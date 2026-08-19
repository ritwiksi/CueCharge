import pandas as pd
import numpy as np

ANYTIME_DEMAND = 37.37  # Simplified PG&E B-19 assumption for the MVP.


def tariff_arrays(timestamps):
    """Return simplified energy and demand-rate arrays for backtesting."""
    prices, d_rates = [], []
    for ts in timestamps:
        h, m = ts.hour, ts.month
        is_summer = 6 <= m <= 9
        if is_summer:
            if 16 <= h < 21:
                p, d = 0.35, 46.16
            elif 14 <= h < 16 or 21 <= h < 23:
                p, d = 0.22, 10.52
            else:
                p, d = 0.15, 0.0
        else:
            if 16 <= h < 21:
                p, d = 0.22, 2.31
            else:
                p, d = 0.14, 0.0
        prices.append(p)
        d_rates.append(d)
    return np.array(prices), np.array(d_rates), []


def compute_bill(timestamps, net_kw):
    """Compute the MVP's simplified modeled bill from 15-minute grid load."""
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps), "grid": np.maximum(net_kw, 0)})
    df["month"] = df["ts"].dt.to_period("M")
    e_p, d_p, _ = tariff_arrays(df["ts"])
    df["e_p"], df["d_p"] = e_p, d_p

    energy_cost = (df["grid"] * 0.25 * df["e_p"]).sum()
    anytime_demand = df.groupby("month")["grid"].max().sum() * ANYTIME_DEMAND

    period_demand = 0
    df_p = df[df["d_p"] > 0]
    if not df_p.empty:
        for _, g in df_p.groupby(["month", "d_p"]):
            period_demand += g["grid"].max() * g["d_p"].iloc[0]

    return energy_cost + anytime_demand + period_demand
