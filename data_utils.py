import pandas as pd
import numpy as np
import requests
import glob
import os

def get_real_data_stack(building_column="WarehouseNew2004", days=30):
    # 1. Load Building Data (From your local CSV)
    csv_files = glob.glob("sf_building_profiles_lite.csv")
    if not csv_files:
        raise FileNotFoundError("sf_building_profiles.csv missing. Please ensure it's in the root folder.")
    
    df_bldg = pd.read_csv(csv_files[0])
    df_bldg.columns = df_bldg.columns.str.strip()
    load_kw = np.array(df_bldg[building_column].values[:days*24], dtype=float)

    # 2. CALL NREL PVWATTS API (Real Research Data)
    # We use the DEMO_KEY which is public for development.
    # Lat/Lon for San Francisco
    url = "https://developer.nrel.gov/api/pvwatts/v6.json"
    params = {
        'api_key': 'DEMO_KEY',
        'lat': 37.7749,
        'lon': -122.4194,
        'system_capacity': 100, # 100kW system
        'azimuth': 180,
        'tilt': 20,
        'array_type': 1,
        'losses': 14,
        'dataset': 'tmy3',
        'timeframe': 'hourly'
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        # The API returns 8760 hours of real research solar data
        solar_data = r.json()['outputs']['ac']
        solar_kw = np.array(solar_data[:days*24]) / 1000.0 # Convert Watts to kW
    except Exception as e:
        # If API is down, we use a slightly randomized physical model so the sim doesn't crash
        print(f"API Error: {e}. Using backup physical model.")
        hour_of_day = np.tile(np.arange(24), days)
        solar_kw = np.where((hour_of_day >= 6) & (hour_of_day <= 18), 
                            80 * np.sin(np.pi * (hour_of_day - 6) / 12) * np.random.uniform(0.7, 1.0, days*24), 
                            0.0)

    time_index = pd.date_range("2024-01-01", periods=len(load_kw), freq="H")

    return pd.DataFrame({
        "timestamp": time_index,
        "load_kw": load_kw,
        "solar_kw": solar_kw
    })