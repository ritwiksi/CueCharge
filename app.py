import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from engine import RollingBrain, LoadForecaster, fixed_rule_dispatch, persistence_forecast
from data_utils import get_real_data_stack, recommend_bess_size
from tariffs import tariff_arrays, compute_bill, ANYTIME_DEMAND

st.set_page_config(page_title="CueCharge Dispatch", layout="wide")
st.title("CueCharge")
st.caption("Real-data battery dispatch backtest")

with st.sidebar:
    st.header("Simulation")
    building_id = st.text_input("ComStock Building ID", value=os.getenv("COMSTOCK_BLDG_ID", ""))
    state = st.text_input("State", value=os.getenv("COMSTOCK_STATE", "CA")).upper()
    latitude = st.number_input("Latitude", value=float(os.getenv("COMSTOCK_LAT", "37.77")), format="%.5f")
    longitude = st.number_input("Longitude", value=float(os.getenv("COMSTOCK_LON", "-122.41")), format="%.5f")
    pv_capacity = st.number_input("PV Capacity (kW)", min_value=1.0, value=250.0)
    sim_days = st.slider("Backtest Window (Days)", 7, 30, 14)
    execute = st.button("RUN BACKTEST", type="primary")

if execute:
    if not building_id.strip():
        st.error("Enter a real ComStock bldg_id from the 2025 Release 3 metadata.")
        st.stop()

    data = get_real_data_stack(
        building_id.strip(), state, latitude, longitude, sim_days + 21,
        start="2018-01-01", pv_capacity_kw=pv_capacity,
    )

    train_data = data.iloc[: 14 * 96].copy()
    eval_data = data.iloc[14 * 96 : (14 + sim_days) * 96].reset_index(drop=True)
    forecaster = LoadForecaster()

    # Train only on the historical window available before the evaluation period.
    forecaster.train(train_data)
    rec_pwr, rec_cap = recommend_bess_size(train_data["load_kw"])
    cap, pwr = rec_cap, rec_pwr
    brain = RollingBrain(cap, pwr)
    baseline_soc = cap * 0.5
    results = []

    for d_idx in range(sim_days):
        start, end = d_idx * 96, (d_idx + 1) * 96
        day = eval_data.iloc[start:end].copy()
        day_ts = day["timestamp"].reset_index(drop=True)

        b_charge, b_discharge = fixed_rule_dispatch(
            day["load_kw"].values, day["solar_kw"].values, day_ts,
            cap, pwr, baseline_soc,
        )
        baseline_net = (day["load_kw"].values - day["solar_kw"].values + b_charge - b_discharge).clip(min=0)
        baseline_soc = max(
            cap * 0.10,
            min(cap, baseline_soc + b_charge.sum() * 0.25 * 0.93 - b_discharge.sum() * 0.25 / 0.93),
        )

        # Re-optimize every 15 minutes and execute only the first action.
        for i in range(96):
            ts = day_ts.iloc[i]
            history = pd.concat([train_data, eval_data.iloc[: start + i]])
            load_f = forecaster.predict_24h(ts, history)
            solar_f = persistence_forecast("solar_kw", ts, history)
            horizon_ts = pd.date_range(ts, periods=96, freq="15min")
            prices_f, d_rates_f, _ = tariff_arrays(horizon_ts)

            c_plan, d_plan = brain.solve_horizon(
                load_f, solar_f, prices_f, d_rates_f.max() + ANYTIME_DEMAND,
            )
            c, d = float(c_plan[0]), float(d_plan[0])
            actual_net = max(day["load_kw"].iloc[i] - day["solar_kw"].iloc[i] + c - d, 0)
            brain.update_state(c, d, actual_net)
            results.append({"Timestamp": ts, "FixedRule_Net": baseline_net[i], "CueCharge_Net": actual_net})

    res_df = pd.DataFrame(results)
    bill_baseline = compute_bill(res_df["Timestamp"], res_df["FixedRule_Net"])
    bill_cue = compute_bill(res_df["Timestamp"], res_df["CueCharge_Net"])
    savings = bill_baseline - bill_cue
    savings_pct = (savings / bill_baseline * 100) if bill_baseline else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Fixed Rule", f"${bill_baseline:,.0f}")
    c2.metric("CueCharge", f"${bill_cue:,.0f}")
    c3.metric("Savings", f"${savings:,.0f}", f"{savings_pct:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_df["Timestamp"], y=res_df["FixedRule_Net"], name="Fixed Rule"))
    fig.add_trace(go.Scatter(x=res_df["Timestamp"], y=res_df["CueCharge_Net"], name="CueCharge"))
    fig.update_layout(height=500, yaxis_title="Grid Load (kW)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Savings = Fixed Rule bill − CueCharge bill.")
