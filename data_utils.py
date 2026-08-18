import pandas as pd
import numpy as np
import requests
import glob
import os

try:
    import streamlit as st
except ImportError:
    st = None


def _get_nlr_api_key():
    """Pulls from Streamlit secrets first, then env var, then falls back to DEMO_KEY."""
    if st is not None:
        try:
            return st.secrets["NLR_API_KEY"]
        except Exception:
            pass
    return os.environ.get("NLR_API_KEY", "DEMO_KEY")


def get_real_data_stack(building_column="WarehouseNew2004", days=30, lat=37.7749, lon=-122.4194):
    csv_files = glob.glob("sf_building_profiles_lite.csv")
    if not csv_files:
        raise FileNotFoundError("sf_building_profiles_lite.csv missing. Please ensure it's in the root folder.")

    df_bldg = pd.read_csv(csv_files[0])
    df_bldg.columns = df_bldg.columns.str.strip()
    load_kw = np.array(df_bldg[building_column].values[:days * 24], dtype=float)

    # Size the solar array off THIS building's own peak load instead of a hardcoded 100kW,
    # so a Hospital and a QSR don't get an identical solar profile.
    system_capacity = max(50.0, float(np.max(load_kw)) * 1.2)

    url = "https://developer.nlr.gov/api/pvwatts/v8.json"
    params = {
        'api_key': _get_nlr_api_key(),
        'lat': lat, 'lon': lon,
        'system_capacity': system_capacity,
        'azimuth': 180, 'tilt': 20, 'array_type': 1, 'module_type': 0, 'losses': 14,
        'dataset': 'nsrdb', 'timeframe': 'hourly'
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        solar_data = r.json()['outputs']['ac']
        solar_kw = np.array(solar_data[:days * 24]) / 1000.0
    except Exception as e:
        print(f"API Error: {e}. Using backup physical model.")
        hour_of_day = np.tile(np.arange(24), days)
        solar_kw = np.where(
            (hour_of_day >= 6) & (hour_of_day <= 18),
            (system_capacity * 0.8) * np.sin(np.pi * (hour_of_day - 6) / 12) * np.random.uniform(0.7, 1.0, days * 24),
            0.0
        )

    time_index = pd.date_range("2024-01-01", periods=len(load_kw), freq="h")
    return pd.DataFrame({"timestamp": time_index, "load_kw": load_kw, "solar_kw": solar_kw})