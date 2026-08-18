import pandas as pd
import numpy as np

ANYTIME_DEMAND = 37.37  # $/kW, applies to each month's overall peak regardless of period


def get_b19_details(dt):
    """
    Returns (energy_price, period_demand_charge, period_name) based on PG&E B-19.
    period_demand_charge is IN ADDITION to ANYTIME_DEMAND above (matches B-19's stacked
    anytime + peak-period demand structure).
    """
    month, hour = dt.month, dt.hour
    is_summer = 6 <= month <= 9

    if is_summer:
        if 16 <= hour < 21:
            return 0.18648, 46.16, "Peak"
        if 14 <= hour < 16 or 21 <= hour < 23:
            return 0.14775, 10.52, "Part-Peak"
        return 0.12037, 0.0, "Off-Peak"
    else:
        if 16 <= hour < 21:
            return 0.16188, 2.31, "Peak"
        # Super Off-Peak (9am-2pm) is Mar/Apr/May only per B-19 -- Oct-Feb at those hours
        # is regular Off-Peak. Previous version wrongly applied it to all winter months.
        if 9 <= hour < 14 and month in (3, 4, 5):
            return 0.06442, 0.0, "Super Off-Peak"
        return 0.12026, 0.0, "Off-Peak"


def tariff_arrays(timestamps):
    """Returns energy_price, period_demand_rate, period_name arrays for a list/Series of datetimes."""
    rows = [get_b19_details(pd.Timestamp(t)) for t in timestamps]
    energy = np.array([r[0] for r in rows])
    demand = np.array([r[1] for r in rows])
    period = [r[2] for r in rows]
    return energy, demand, period


def demand_incentive(timestamps):
    """
    Combined per-hour demand-cost SIGNAL for the optimizer (not for billing).
    The anytime component (ANYTIME_DEMAND) applies to the month's peak draw no matter
    what hour it happens, so the optimizer needs that cost pressure active every hour --
    not just during Peak/Part-Peak. Without this, off-peak hours look "free" to the LP
    even though a new peak built off-peak still gets billed at ANYTIME_DEMAND.
    """
    _, period_rate, _ = tariff_arrays(timestamps)
    return (period_rate + ANYTIME_DEMAND) / 20


def compute_bill(timestamps, net_grid_kw):
    """
    Real B-19-style monthly bill: energy charge + monthly anytime demand + monthly
    period-specific demand (Peak/Part-Peak), each billed on that period's own max draw.
    Export (negative net_grid) is treated as uncompensated, matching no-NEM assumption.
    """
    df = pd.DataFrame({"ts": pd.to_datetime(pd.Series(timestamps).values),
                        "grid": np.maximum(np.asarray(net_grid_kw), 0)})
    energy_price, demand_rate, period = tariff_arrays(df["ts"])
    df["energy_price"], df["demand_rate"], df["period"] = energy_price, demand_rate, period
    df["month"] = df["ts"].dt.to_period("M")

    energy_cost = (df["grid"] * df["energy_price"]).sum()
    anytime_cost = df.groupby("month")["grid"].max().sum() * ANYTIME_DEMAND

    period_cost = 0.0
    priced = df[df["demand_rate"] > 0]
    for (_, _), g in priced.groupby(["month", "period"]):
        period_cost += g["grid"].max() * g["demand_rate"].iloc[0]

    return energy_cost + anytime_cost + period_cost