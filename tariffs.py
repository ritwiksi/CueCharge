import pandas as pd

def get_b19_details(dt):
    """
    Returns (energy_price, demand_charge, period_name) based on PG&E B-19.
    """
    month = dt.month
    hour = dt.hour
    is_summer = 6 <= month <= 9
    
    # Anytime Demand Charge (The 'Maximum' charge)
    anytime_demand = 37.37 
    
    if is_summer:
        # Energy Prices
        if 16 <= hour < 21: 
            return 0.18648, 46.16, "Peak"        # Max Peak Demand also applies here
        if 14 <= hour < 16 or 21 <= hour < 23: 
            return 0.14775, 10.52, "Part-Peak"
        return 0.12037, 0.0, "Off-Peak"
    else:
        # Winter
        if 16 <= hour < 21: 
            return 0.16188, 2.31, "Peak"
        if 9 <= hour < 14: 
            return 0.06442, 0.0, "Super Off-Peak"
        return 0.12026, 0.0, "Off-Peak"