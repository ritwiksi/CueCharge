import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine import RollingBrain
from data_utils import get_real_data_stack
from tariffs import tariff_arrays, compute_bill, demand_incentive

# 1. INDUSTRIAL AESTHETICS SETUP
st.set_page_config(page_title="CueCharge Dispatch Node", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: #E2E8F0; }
    .stNumberInput input { background-color: #1E293B !important; color: white !important; border: 1px solid #334155 !important; }
    .stSelectbox div { background-color: #1E293B !important; }
    .metric-card {
        background: #1E293B;
        padding: 24px;
        border-radius: 4px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background-color: #2563EB !important;
        color: white !important;
        width: 100%;
        border-radius: 2px;
        height: 3.5em;
        font-weight: bold;
        border: none;
    }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; color: #F8FAFC; }
    </style>
    """, unsafe_allow_html=True)

st.title("CueCharge: Industrial Asset Dispatch")
st.caption("Quantitative Simulation | PG&E B-19 Secondary (live TOU + tiered demand tariff) | NREL PVWatts")

# 2. SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("Asset Configuration")
    facility = st.selectbox("Facility Profile", [
        "WarehouseNew2004",
        "QuickServiceRestaurantNew2004",
        "SuperMarketNew2004",
        "HospitalNew2004",
        "SecondarySchoolNew2004",
        "LargeOfficeNew2004"
    ])

    cap = st.number_input("BESS Capacity (kWh)", value=200.0, step=10.0)
    pwr = st.number_input("BESS Power (kW)", value=100.0, step=10.0)
    sim_days = st.number_input("Simulation Window (Days)", value=30, min_value=2, max_value=60)

    st.markdown("---")
    st.caption("Energy & demand pricing pulled live from PG&E B-19 (seasonal TOU + anytime/peak/part-peak demand). No manual price entry.")
    execute = st.button("EXECUTE DISPATCH STRATEGY")


def simulate_fixed_rule(data, cap, pwr, eta=0.93):
    """
    Naive static-rule battery: charge overnight, discharge during the evening peak window.
    This -- not 'no battery' -- is CueCharge's real competitive baseline, since it's what
    an unmanaged battery already does today.
    """
    soc = cap * 0.5
    net = []
    for _, row in data.iterrows():
        hour = row['timestamp'].hour
        load, solar = row['load_kw'], row['solar_kw']
        c = d = 0.0
        if 0 <= hour < 6:
            c = min(pwr, max(0.0, (cap - soc) / eta))
        elif 16 <= hour < 21:
            d = min(pwr, soc * eta, max(load - solar, 0))
        soc = max(0, min(cap, soc + c * eta - d / eta))
        net.append(load - solar + c - d)
    return np.array(net)


# 3. SIMULATION ENGINE
if execute:
    with st.spinner("CALCULATING TOTAL COST OF OWNERSHIP..."):
        try:
            data = get_real_data_stack(facility, sim_days)
        except Exception as e:
            st.error(f"Data Fetch Failure: {e}")
            st.stop()

        brain = RollingBrain(cap, pwr)
        results = []
        current_period = None

        for t in range(24, len(data)):
            ts_now = data.iloc[t]['timestamp']
            month_now = ts_now.to_period("M")
            if current_period is not None and month_now != current_period:
                # New billing month -- the real anytime/period demand charges reset here,
                # so the optimizer's internal ratchet has to reset too, or it thinks last
                # month's peak was already "paid for" this month.
                brain.peak_so_far = 0.0
            current_period = month_now

            yesterday_load = data['load_kw'].iloc[t - 24:t].values
            lookahead = min(24, len(data) - t)
            solar_forecast = data['solar_kw'].iloc[t:t + lookahead].values
            if len(solar_forecast) < 24:
                solar_forecast = np.pad(solar_forecast, (0, 24 - len(solar_forecast)), 'constant')

            # Real hourly B-19 forecast instead of a flat, user-typed number
            future_ts = data['timestamp'].iloc[t:t + lookahead]
            price_forecast, _, _ = tariff_arrays(future_ts)
            incentive_forecast = demand_incentive(future_ts)
            if len(price_forecast) < 24:
                price_forecast = np.pad(price_forecast, (0, 24 - len(price_forecast)), 'edge')

            # Use the WORST-CASE rate in the lookahead window, not just the current hour's --
            # otherwise a 3am solve prices a peak it creates later that day (during Peak
            # hours) using the cheap 3am rate, which underprices the real risk.
            c, d = brain.solve_best_move(yesterday_load, solar_forecast, price_forecast, incentive_forecast.max())

            actual_l, actual_s = data.iloc[t]['load_kw'], data.iloc[t]['solar_kw']
            net_grid = actual_l - actual_s + c - d
            brain.update_state(c, d, net_grid)

            results.append({
                "Timestamp": data.iloc[t]['timestamp'],
                "Baseline_Load": actual_l,
                "Net_Grid_Draw": net_grid,
                "SoC": brain.soc
            })

        df_res = pd.DataFrame(results)
        sim_window = data.iloc[24:].reset_index(drop=True)

        # THREE-WAY COMPARISON: no battery / naive fixed-rule battery / CueCharge
        no_batt_net = np.maximum(sim_window['load_kw'] - sim_window['solar_kw'], 0)
        fixed_net = simulate_fixed_rule(sim_window, cap, pwr)

        cost_no_battery = compute_bill(sim_window['timestamp'], no_batt_net)
        cost_fixed_rule = compute_bill(sim_window['timestamp'], fixed_net)
        cost_optimized = compute_bill(df_res['Timestamp'], df_res['Net_Grid_Draw'])

        savings_vs_fixed_rule = cost_fixed_rule - cost_optimized  # the number that proves dispatch intelligence

        # 4. ANALYTICS & VISUALIZATION
        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, color in [
            (m1, "NO-BATTERY BILL", cost_no_battery, "#F8FAFC"),
            (m2, "FIXED-RULE BILL", cost_fixed_rule, "#F8FAFC"),
            (m3, "CUECHARGE BILL", cost_optimized, "#F8FAFC"),
            (m4, "SAVINGS VS. FIXED-RULE", savings_vs_fixed_rule, "#4ADE80"),
        ]:
            with col:
                st.markdown(f"""<div class="metric-card">
                    <small style='color: #94A3B8;'>{label}</small><br>
                    <span style='font-size: 22px; font-weight: bold; color:{color};'>${val:,.2f}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("### Dispatch Strategy Performance")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_res['Timestamp'], y=df_res['Baseline_Load'],
            name="Raw Building Load", line=dict(color='#64748B', width=1, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=sim_window['timestamp'], y=fixed_net,
            name="Fixed-Rule Battery Draw", line=dict(color='#F59E0B', width=1.5, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df_res['Timestamp'], y=df_res['Net_Grid_Draw'],
            name="CueCharge Optimized Draw", line=dict(color='#3B82F6', width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=df_res['Timestamp'], y=df_res['SoC'],
            name="Battery SoC (kWh)", line=dict(color='#0EA5E9', width=1),
            yaxis="y2", fill='tozeroy', fillcolor='rgba(14, 165, 233, 0.1)'
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=600,
            hovermode="x unified",
            yaxis=dict(title="Power (kW)", gridcolor='#334155'),
            yaxis2=dict(title="SoC (kWh)", overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Bill math is full B-19: seasonal TOU energy + monthly anytime demand + peak/part-peak period demand.")
else:
    st.info("Configure asset specifications and click 'Execute Dispatch Strategy'.")