import glob
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    NREL_API_KEY = st.secrets["NREL_API_KEY"]
except Exception:
    NREL_API_KEY = os.getenv("NREL_API_KEY")


def _pvwatts_solar_hourly(system_cap, days):
    """Return the July TMY3 solar slice used by the legacy fallback."""
    if not NREL_API_KEY:
        return None

    response = requests.get(
        "https://developer.nrel.gov/api/pvwatts/v8.json",
        params={
            "api_key": NREL_API_KEY,
            "lat": 37.77,
            "lon": -122.41,
            "system_capacity": system_cap,
            "azimuth": 180,
            "tilt": 20,
            "array_type": 1,
            "module_type": 0,
            "losses": 14,
            "dataset": "tmy3",
            "timeframe": "hourly",
        },
        timeout=10,
    )
    response.raise_for_status()
    year = np.asarray(response.json()["outputs"]["ac"], dtype=float)
    july_start = 181 * 24
    return year[july_start : july_start + days * 24]


def _load_comstock_15min(building_column, days):
    """Load a local 15-minute ComStock/EULP export if one is available.

    Expected format: a CSV or parquet file with a timestamp column and one or
    more building-load columns in kW. Set COMSTOCK_15MIN_PATH to the file.
    """
    path = os.getenv("COMSTOCK_15MIN_PATH")
    if not path:
        for candidate in ["comstock_15min.csv", "comstock_15min.parquet"]:
            if os.path.exists(candidate):
                path = candidate
                break
    if not path or not os.path.exists(path):
        return None

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    timestamp_col = next((c for c in ["timestamp", "Timestamp", "time", "Time"] if c in df.columns), None)
    if timestamp_col is None or building_column not in df.columns:
        raise ValueError(
            "15-minute ComStock file must contain a timestamp column and the selected building column."
        )

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[timestamp_col]),
            "load_kw": pd.to_numeric(df[building_column], errors="coerce"),
        }
    ).dropna()
    out = out.sort_values("timestamp").drop_duplicates("timestamp")
    out = out.set_index("timestamp").resample("15min").mean().interpolate(limit=4).reset_index()
    out = out.iloc[: days * 96].copy()
    if len(out) < days * 96:
        raise ValueError("15-minute ComStock file does not contain enough data for this backtest")
    return out


def get_real_data_stack(building_column, days):
    """Build the backtest data stack.

    Preferred path: a public 15-minute ComStock/EULP export supplied locally.
    Legacy path: the compact hourly profile plus PVWatts, retained only so the
    app remains runnable while the public 15-minute export is being downloaded.
    """
    comstock = _load_comstock_15min(building_column, days)
    if comstock is not None:
        # Solar remains a separate input until the historical NSRDB adapter is added.
        system_cap = max(float(comstock["load_kw"].max()) * 0.5, 10.0)
        solar_hourly = None
        try:
            solar_hourly = _pvwatts_solar_hourly(system_cap, int(np.ceil(days)))
        except (requests.RequestException, KeyError, ValueError):
            pass
        if solar_hourly is None or len(solar_hourly) < days * 24:
            solar_hourly = np.zeros(days * 24)
        solar_index = pd.date_range(comstock["timestamp"].iloc[0].floor("h"), periods=len(solar_hourly), freq="h")
        solar_df = pd.DataFrame({"timestamp": solar_index, "solar_kw": solar_hourly})
        comstock["hour"] = comstock["timestamp"].dt.floor("h")
        comstock = comstock.merge(solar_df, left_on="hour", right_on="timestamp", how="left", suffixes=("", "_solar"))
        comstock["solar_kw"] = comstock["solar_kw"].fillna(0.0)
        return comstock[["timestamp", "load_kw", "solar_kw"]].reset_index(drop=True)

    # Legacy fallback only. This is deliberately deterministic and clearly separated
    # from the real 15-minute ComStock path above.
    csv_files = glob.glob("sf_building_profiles_lite.csv")
    if not csv_files:
        raise FileNotFoundError(
            "No 15-minute ComStock file found. Add comstock_15min.csv/parquet or set COMSTOCK_15MIN_PATH."
        )

    df_bldg = pd.read_csv(csv_files[0])
    df_bldg.columns = df_bldg.columns.str.strip()
    if building_column not in df_bldg.columns:
        raise ValueError(f"Unknown building profile: {building_column}")

    load_hourly = df_bldg[building_column].astype(float).values[: days * 24]
    if len(load_hourly) < days * 24:
        raise ValueError("Building profile does not contain enough hourly data")

    system_cap = max(float(np.max(load_hourly)) * 0.5, 10.0)
    try:
        solar_hourly = _pvwatts_solar_hourly(system_cap, days)
    except (requests.RequestException, KeyError, ValueError):
        solar_hourly = None

    if solar_hourly is None or len(solar_hourly) != days * 24:
        hours = np.arange(days * 24) % 24
        daylight = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None)
        solar_hourly = system_cap * daylight

    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-07-01", periods=days * 24, freq="h")
    rows = []
    variance_coeff = {
        "WarehouseNew2004": 0.42,
        "HospitalNew2004": 0.12,
        "SuperMarketNew2004": 0.22,
        "QuickServiceRestaurantNew2004": 0.38,
    }.get(building_column, 0.20)

    for i, h_load in enumerate(load_hourly):
        raw_spikes = rng.lognormal(mean=0, sigma=variance_coeff, size=4)
        quarter_loads = (raw_spikes / raw_spikes.sum()) * h_load * 4
        for j, quarter_load in enumerate(quarter_loads):
            rows.append(
                {
                    "timestamp": timestamps[i] + pd.Timedelta(minutes=15 * j),
                    "load_kw": quarter_load,
                    "solar_kw": solar_hourly[i],
                }
            )

    return pd.DataFrame(rows)


def recommend_bess_size(load):
    peak = float(np.max(load))
    pwr = max(round(peak * 0.4, -1), 10.0)
    cap = pwr * 2
    return pwr, cap
