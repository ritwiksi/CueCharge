import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine import RollingBrain, LoadForecaster, fixed_rule_dispatch
from data_utils import get_real_data_stack, recommend_bess_size
from tariffs import tariff_arrays, compute_bill, ANYTIME_DEMAND

st.set_page_config(page_title="CueCharge Dispatch", layout="wide")
st.title("CueCharge: Battery Dispatch Backtest")
st.caption("Historical simulation only — not a live battery controller.")

with st.sidebar:
    st.header("Site")
    facility = st.selectbox(
        "Building Profile",
        ["WarehouseNew2004", "HospitalNew2004", "SuperMarketNew2004", "QuickServiceRestaurantNew2004"],
    )
    init_data = get_real_data_stack(facility, 1)
    rec_pwr, rec_cap = recommend_bess_size(init_data["load_kw"])
    st.info(f"Peak: {np.max(init_data['load_kw']):.1f} kW | Suggested BESS: {rec_pwr:.0f} kW / {rec_cap:.0f} kWh")
    cap = st.number_input("BESS Capacity (kWh)", min_value=1.0, value=float(rec_cap))
    pwr = st.number_input("BESS Power (kW)", min_value=1.0, value=float(rec_pwr))
    sim_days = st.slider("Backtest Window (Days)", 7, 30, 14)
    execute = st.button("RUN BACKTEST")

if execute:
    data = get_real_data_stack(facility, sim_days + 14)
    train_data = data.iloc[: 14 * 96].copy()
    eval_data = data.iloc[14 * 96 : (14 + sim_days) * 96].reset_index(drop=True)

    forecaster = LoadForecaster()
    forecaster.train(train_data)
    brain = RollingBrain(cap, pwr)
    baseline_soc = cap * 0.5
    results = []

    for d_idx in range(sim_days):
        start, end = d_idx * 96, (d_idx + 1) * 96
        day = eval_data.iloc[start:end].copy()
        ts_start = day.iloc[0]["timestamp"]
        history = pd.concat([train_data, eval_data.iloc[:start]]).tail(14 * 96)

        load_f = forecaster.predict_24h(ts_start, history)
        prices_f, d_rates_f, _ = tariff_arrays(day["timestamp"])

        # Weather/solar forecast is deliberately conservative for this MVP:
        # use the same historical solar profile rather than future measured solar.
        solar_f = day["solar_kw"].values

        # Fixed-rule battery baseline: no forecasting, no optimization.
        b_charge, b_discharge = fixed_rule_dispatch(
            day["load_kw"].values,
            day["solar_kw"].values,
            day["timestamp"],
            cap,
            pwr,
            baseline_soc,
        )
        baseline_net = np.maximum(day["load_kw"].values - day["solar_kw"].values + b_charge - b_discharge, 0)

        # CueCharge optimization. VPP revenue is intentionally excluded from the MVP.
        c_charge, c_discharge = brain.solve_full_day(
            load_f,
            solar_f,
            prices_f,
            d_rates_f.max() + ANYTIME_DEMAND,
        )

        for i in range(96):
            actual_net = day.iloc[i]["load_kw"] - day.iloc[i]["solar_kw"] + c_charge[i] - c_discharge[i]
            actual_net = max(actual_net, 0)
            brain.update_state(c_charge[i], c_discharge[i], actual_net)
            results.append(
                {
                    "Timestamp": day.iloc[i]["timestamp"],
                    "Raw_Net": max(day.iloc[i]["load_kw"] - day.iloc[i]["solar_kw"], 0),
                    "Baseline_Net": baseline_net[i],
                    "CueCharge_Net": actual_net,
                    "Baseline_Charge": b_charge[i],
                    "Baseline_Discharge": b_discharge[i],
                    "CueCharge_Charge": c_charge[i],
                    "CueCharge_Discharge": c_discharge[i],
                    "SoC": brain.soc,
                }
            )

    res_df = pd.DataFrame(results)
    bill_unmanaged = compute_bill(res_df["Timestamp"], res_df["Raw_Net"])
    bill_baseline = compute_bill(res_df["Timestamp"], res_df["Baseline_Net"])
    bill_cue = compute_bill(res_df["Timestamp"], res_df["CueCharge_Net"])

    baseline_savings = bill_unmanaged - bill_baseline
    cue_savings = bill_unmanaged - bill_cue
    incremental_savings = bill_baseline - bill_cue

    st.subheader("Financial Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unmanaged", f"${bill_unmanaged:,.0f}")
    c2.metric("Fixed Rule", f"${bill_baseline:,.0f}", f"-${baseline_savings:,.0f}")
    c3.metric("CueCharge", f"${bill_cue:,.0f}", f"-${cue_savings:,.0f}")
    c4.metric("Incremental Value", f"${incremental_savings:,.0f}")

    st.caption("The key metric is Incremental Value: CueCharge savings versus a customer who already owns and operates a battery with a fixed rule.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_df["Timestamp"], y=res_df["Raw_Net"], name="Unmanaged"))
    fig.add_trace(go.Scatter(x=res_df["Timestamp"], y=res_df["Baseline_Net"], name="Fixed Rule"))
    fig.add_trace(go.Scatter(x=res_df["Timestamp"], y=res_df["CueCharge_Net"], name="CueCharge"))
    fig.update_layout(template="plotly_dark", height=500, yaxis_title="Grid Load (kW)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Important modeling caveats")
    st.write(
        "Building profiles are hourly and reconstructed to 15-minute resolution for this MVP. "
        "Solar is based on a PVWatts TMY3 simulation, not measured site weather. "
        "The demand-charge implementation is a simplified tariff model and should not be presented as an official utility bill."
    )
