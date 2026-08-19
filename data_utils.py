import os
import glob
import numpy as np
import pandas as pd
import requests
import streamlit as st


try:
    NREL_API_KEY = st.secrets["NREL_API_KEY"]
except Exception:
    NREL_API_KEY = os.getenv("NREL_API_KEY")


def _pvwatts_solar_hourly(system_cap, days):
    """Return a July TMY3 solar slice aligned to our July simulation dates."""
    if not NREL_API_KEY:
        return None

    url = "https://developer.nrel.gov/api/pvwatts/v8.json"
    params = {
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
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    outputs = response.json()["outputs"]
    year = np.asarray(outputs["ac"], dtype=float)

    # TMY3 is an annual representative year, not a July-only response.
    # July starts at hour 434 in a non-leap year.
    july_start = 181 * 24
    return year[july_start:july_start + days * 24] / 1000.0


def get_real_data_stack(building_column, days):
    """
    Build a reproducible simulation dataset from hourly building profiles.

    The building profiles are hourly. We reconstruct four 15-minute intervals
    only so the demand-charge model has sub-hourly resolution. These are
    synthetic sub-hourly values, NOT real 15-minute meter telemetry.
    """
    csv_files = glob.glob("sf_building_profiles_lite.csv")
    if not csv_files:
        raise FileNotFoundError("sf_building_profiles_lite.csv not found")

    df_bldg = pd.read_csv(csv_files[0])
    df_bldg.columns = df_bldg.columns.str.strip()
    if building_column not in df_bldg.columns:
        raise ValueError(f"Unknown building profile: {building_column}")

    load_hourly = df_bldg[building_column].astype(float).values[: days * 24]
    if len(load_hourly) < days * 24:
        raise ValueError("Building profile does not contain enough hourly data")

    # Keep solar assumptions explicit rather than silently pretending the
    # weather API is measured site telemetry.
    system_cap = max(float(np.max(load_hourly)) * 0.5, 10.0)
    try:
        solar_hourly = _pvwatts_solar_hourly(system_cap, days)
    except (requests.RequestException, KeyError, ValueError):
        solar_hourly = None

    if solar_hourly is None or len(solar_hourly) != days * 24:
        hours = np.arange(days * 24) % 24
        daylight = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None)
        solar_hourly = system_cap * daylight

    variance_coeff = {
        "WarehouseNew2004": 0.42,
        "HospitalNew2004": 0.12,
        "SuperMarketNew2004": 0.22,
        "QuickServiceRestaurantNew2004": 0.38,
    }.get(building_column, 0.20)

    # Fixed seed makes every backtest reproducible.
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-07-01", periods=days * 24, freq="h")

    rows = []
    for i, h_load in enumerate(load_hourly):
        raw_spikes = rng.lognormal(mean=0, sigma=variance_coeff, size=4)
        # Preserve hourly energy while introducing synthetic 15-min shape.
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
