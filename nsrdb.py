import io
import os

import numpy as np
import pandas as pd
import requests


NSRDB_URL = "https://developer.nlr.gov/api/nsrdb/v2/solar/psm3-5min-download.csv"


def fetch_nsrdb_5min(latitude, longitude, year=2018, api_key=None, email=None):
    """Download historical NSRDB PSM3 5-minute data for one site/year."""
    api_key = api_key or os.getenv("NLR_API_KEY") or os.getenv("NREL_API_KEY")
    email = email or os.getenv("NSRDB_EMAIL")
    if not api_key:
        raise RuntimeError("NLR_API_KEY is required for historical NSRDB data.")
    if not email:
        raise RuntimeError("NSRDB_EMAIL is required for historical NSRDB data.")

    params = {
        "api_key": api_key,
        "wkt": f"POINT({float(longitude)} {float(latitude)})",
        "names": str(year),
        "interval": 5,
        "utc": "false",
        "leap_day": "false",
        "full_name": "CueCharge",
        "email": email,
        "affiliation": "CueCharge MVP",
        "reason": "Academic research",
        "mailing_list": "false",
        "attributes": "ghi,dhi,dni,air_temperature,wind_speed",
    }

    response = requests.get(NSRDB_URL, params=params, timeout=120)
    response.raise_for_status()
    text = response.text

    # NSRDB CSV has several metadata rows before the tabular header.
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Year,Month,Day,Hour,Minute")), None)
    if header_idx is None:
        raise RuntimeError("NSRDB response did not contain the expected CSV header.")

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    required = {"Year", "Month", "Day", "Hour", "Minute", "GHI", "DHI", "DNI"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"NSRDB response is missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(
        dict(
            year=df["Year"].astype(int),
            month=df["Month"].astype(int),
            day=df["Day"].astype(int),
            hour=df["Hour"].astype(int),
            minute=df["Minute"].astype(int),
        )
    )
    df["ghi"] = pd.to_numeric(df["GHI"], errors="coerce").clip(lower=0)
    df["dhi"] = pd.to_numeric(df["DHI"], errors="coerce").clip(lower=0)
    df["dni"] = pd.to_numeric(df["DNI"], errors="coerce").clip(lower=0)
    return df[["timestamp", "ghi", "dhi", "dni"]].dropna(subset=["timestamp"])


def pv_from_irradiance(nsrdb, system_capacity_kw, tilt_deg=20.0, azimuth_deg=180.0, performance_ratio=0.85):
    """Simple transparent PV conversion from historical irradiance to AC kW.

    Uses GHI as the resource signal for the MVP. System capacity and performance
    ratio are explicit assumptions rather than hidden synthetic data.
    """
    if system_capacity_kw <= 0:
        raise ValueError("system_capacity_kw must be positive")
    out = nsrdb[["timestamp", "ghi"]].copy()
    out["solar_kw"] = np.minimum(
        float(system_capacity_kw),
        float(system_capacity_kw) * out["ghi"] / 1000.0 * float(performance_ratio),
    )
    return out[["timestamp", "solar_kw"]]
