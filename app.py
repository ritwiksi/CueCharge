import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine import RollingBrain, LoadForecaster
from data_utils import get_real_data_stack, recommend_bess_size
from tariffs import tariff_arrays, compute_bill, get_vpp_events, ANYTIME_DEMAND

st.set_page_config(page_title="CueCharge Dispatch", layout="wide")
st.markdown("<style>.stApp { background-color: #0F172A; color: #E2E8F0; }</style>", unsafe_allow_html=True)

st.title("CueCharge: Quantitative Performance Audit")

with st.sidebar:
    st.header("Site Telemetry")
    facility = st.selectbox("Building Profile", ["WarehouseNew2004", "HospitalNew2004", "SuperMarketNew2004", "QuickServiceRestaurantNew2004"])
    
    init_data = get_real_data_stack(facility, 1)
    rec_pwr, rec_cap = recommend_bess_size(init_data['load_kw'])
    
    st.info(f"💡 Site Peak: {np.max(init_data['load_kw']):.1f}kW. Recommended BESS: {rec_pwr}kW/{rec_cap}kWh.")

    cap = st.number_input("BESS Capacity (kWh)", value=float(rec_cap))
    pwr = st.number_input("BESS Power (kW)", value=float(rec_pwr))
    sim_days = st.slider("Backtest Window (Days)", 7, 30, 14)
    execute = st.button("RUN CUECHARGE OPTIMIZATION")

if execute:
    # Fetch July data for real Summer Tariff Stress
    data = get_real_data_stack(facility, sim_days + 14)
    train_data = data.iloc[:14*96]
    eval_data = data.iloc[14*96 : (14 + sim_days)*96].reset_index(drop=True)
    
    with st.spinner("AI is calculating least-cost dispatch paths..."):
        forecaster = LoadForecaster()
        forecaster.train(train_data)
        brain = RollingBrain(cap, pwr)
        vpp_signals = get_vpp_events(eval_data['timestamp'])
        
        results = []
        num_days = len(eval_data) // 96
        
        for d_idx in range(num_days):
            start, end = d_idx * 96, (d_idx + 1) * 96
            day_slice = eval_data.iloc[start:end]
            ts_start = day_slice.iloc[0]['timestamp']
            
            # 1. AI Forecast
            history = pd.concat([train_data, eval_data.iloc[:start]]).tail(168*4)
            load_f = forecaster.predict_24h(ts_start, history)
            
            # 2. Market Signals (Summer TOU + Demand)
            prices_f, d_rates_f, _ = tariff_arrays(day_slice['timestamp'])
            solar_f = day_slice['solar_kw'].values
            vpp_f = vpp_signals[start:end]
            
            # 3. OPTIMIZE
            c_plan, d_plan = brain.solve_full_day(load_f, solar_f, prices_f, d_rates_f.max() + ANYTIME_DEMAND, vpp_f)
            
            # 4. EXECUTE
            for i in range(96):
                t_idx = start + i
                act_l, act_s = eval_data.iloc[t_idx]['load_kw'], eval_data.iloc[t_idx]['solar_kw']
                
                # The Actual Result
                act_net = act_l - act_s + c_plan[i] - d_plan[i]
                
                brain.update_state(c_plan[i], d_plan[i], act_net)
                results.append({
                    "Timestamp": eval_data.iloc[t_idx]['timestamp'],
                    "Raw_Net": act_l - act_s,
                    "CueCharge_Net": act_net,
                    "VPP_Revenue": (d_plan[i] * 0.25) * vpp_signals[t_idx],
                    "SoC": brain.soc
                })

        res_df = pd.DataFrame(results)
        bill_std = compute_bill(res_df['Timestamp'], res_df['Raw_Net'])
        bill_cue = compute_bill(res_df['Timestamp'], res_df['CueCharge_Net'])
        vpp_earn = res_df['VPP_Revenue'].sum()
        total_value = (bill_std - bill_cue) + vpp_earn

        st.subheader("Financial Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unmanaged Bill", f"${bill_std:,.2f}")
        c2.metric("CueCharge Bill", f"${bill_cue:,.2f}", f"-${bill_std-bill_cue:,.2f}")
        c3.metric("Grid Revenue (VPP)", f"${vpp_earn:,.2f}")
        c4.metric("Total Value Added", f"${total_value:,.2f}")

        # Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_df['Timestamp'], y=res_df['Raw_Net'], name="Standard Load (Solar)", line=dict(color='#64748B', width=1)))
        fig.add_trace(go.Scatter(x=res_df['Timestamp'], y=res_df['CueCharge_Net'], name="CueCharge Managed", line=dict(color='#3B82F6', width=2)))
        fig.update_layout(template="plotly_dark", height=500, yaxis=dict(title="kW"))
        st.plotly_chart(fig, use_container_width=True)