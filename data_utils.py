import io
import os

import numpy as np
import pandas as pd
import requests

from nsrdb import fetch_nsrdb_5min, pv_from_irradiance


COMSTOCK_BASE = (
    "s3://oedi-data-lake/nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/2025/comstock_amy2018_release_3"
)


def _comstock_path(building_id, state, upgrade=0):
    return (
        f"{COMSTOCK_BASE}/timeseries_individual_buildings/by_state/"
        f"upgrade={upgrade}/state={state}/{building_id}-{upgrade}.parquet"
    )


def load_comstock_15min(building_id, state, days, start="2018-01-01"):
    """Load one real ComStock AMY2018 individual-building 15-minute profile.

    ComStock publishes individual building timeseries as kWh per 15-minute
    interval. CueCharge converts that interval energy to average kW.
    """
    try:
        import s3fs
    except ImportError as exc:
        raise RuntimeError("Install s3fs to read the public ComStock OEDI data lake.") from exc

    path = _comstock_path(str(building_id), str(state).upper())
    fs = s3fs.S3FileSystem(anon=True)
    if not fs.exists(path):
        raise FileNotFoundError(
            f"ComStock building file was not found: {path}. "
            "Use a valid bldg_id/state from the 2025 Release 3 metadata."
        )

    table = pd.read_parquet(path, columns=["timestamp", "out.electricity.total.energy_consumption"], filesystem=fs)
    if "out.electricity.total.energy_consumption" not in table.columns:
        raise ValueError("ComStock file does not contain total electricity consumption.")

    out = table.rename(columns={"out.electricity.total.energy_consumption": "interval_kwh"})
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["interval_kwh"] = pd.to_numeric(out["interval_kwh"], errors="coerce")
    out = out.dropna(subset=["timestamp", "interval_kwh"]).sort_values("timestamp")
    out = out.drop_duplicates("timestamp")
    out["load_kw"] = out["interval_kwh"] / 0.25

    start_ts = pd.Timestamp(start)
    out = out[out["timestamp"] >= start_ts].iloc[: days * 96].copy()
    if len(out) < days * 96:
        raise ValueError(f"ComStock building has only {len(out)} usable 15-minute intervals after {start_ts.date()}; need {days * 96}.")
    return out[["timestamp", "load_kw"]].reset_index(drop=True)


def get_real_data_stack(building_id, state, latitude, longitude, days, start="2018-01-01", pv_capacity_kw=None):
    """Build CueCharge's real-data stack from public ComStock + historical NSRDB."""
    load = load_comstock_15min(building_id, state, days, start=start)

    nsrdb = fetch_nsrdb_5min(latitude, longitude, year=pd.Timestamp(start).year)
    solar = pv_from_irradiance(
        nsrdb,
        system_capacity_kw=pv_capacity_kw or max(float(load["load_kw"].max()) * 0.5, 10.0),
    )
    solar["timestamp"] = pd.to_datetime(solar["timestamp"])
    solar = solar.set_index("timestamp").resample("15min").mean().reset_index()

    out = load.merge(solar, on="timestamp", how="left")
    out["solar_kw"] = out["solar_kw"].fillna(0.0)
    if len(out) < days * 96:
        raise ValueError("Historical NSRDB data does not cover the requested ComStock period.")
    return out.iloc[: days * 96].reset_index(drop=True)


def recommend_bess_size(load):
    peak = float(np.max(load))
    pwr = max(round(peak * 0.4, -1), 10.0)
    cap = pwr * 2
    return pwr, cap
