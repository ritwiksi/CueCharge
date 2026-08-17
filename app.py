import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine import RollingBrain
from data_utils import get_real_data_stack

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
st.caption("Quantitative Simulation | PG&E B-19 Secondary | NREL OEDI & PVWatts API")

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
    
    st.header("Market Parameters")
    demand_fee = st.number_input("Anytime Demand Charge ($/kW)", value=37.37)
    energy_price = st.number_input("Flat Energy Price ($/kWh)", value=0.15, format="%.2f")
    sim_days = st.number_input("Simulation Window (Days)", value=30, min_value=1, max_value=365)

    st.markdown("---")
    execute = st.button("EXECUTE DISPATCH STRATEGY")

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

        # Track total energy costs
        base_energy_cost = 0.0
        opt_energy_cost = 0.0

        for t in range(24, len(data)):
            yesterday_load = data['load_kw'].iloc[t-24:t].values
            lookahead = min(24, len(data) - t)
            solar_forecast = data['solar_kw'].iloc[t:t+lookahead].values
            
            if len(solar_forecast) < 24:
                solar_forecast = np.pad(solar_forecast, (0, 24-len(solar_forecast)), 'constant')
            
            # Forecast uses the price the user entered
            price_forecast = np.full(24, energy_price)

            c, d = brain.solve_best_move(yesterday_load, solar_forecast, price_forecast, demand_fee)
            
            actual_l = data.iloc[t]['load_kw']
            actual_s = data.iloc[t]['solar_kw']
            
            # BASELINE: Cost without battery
            net_baseline = max(0, actual_l - actual_s)
            base_energy_cost += net_baseline * energy_price
            
            # OPTIMIZED: Cost with CueCharge
            net_grid = actual_l - actual_s + c - d
            opt_energy_cost += net_grid * energy_price
            
            brain.update_state(c, d, net_grid)
            
            results.append({
                "Timestamp": data.iloc[t]['timestamp'],
                "Baseline_Load": actual_l,
                "Net_Grid_Draw": net_grid,
                "SoC": brain.soc
            })

        df_res = pd.DataFrame(results)
        
        # Demand Charge Calculation
        peak_base = data['load_kw'].max()
        peak_opt = df_res['Net_Grid_Draw'].max()
        
        base_demand_cost = peak_base * demand_fee
        opt_demand_cost = peak_opt * demand_fee
        
        # TOTAL SAVINGS = (Old Total Bill) - (New Total Bill)
        total_baseline = base_energy_cost + base_demand_cost
        total_optimized = opt_energy_cost + opt_demand_cost
        total_savings = total_baseline - total_optimized

        # 4. ANALYTICS & VISUALIZATION
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <small style='color: #94A3B8;'>BASELINE PEAK</small><br>
                <span style='font-size: 24px; font-weight: bold;'>{peak_base:.2f} kW</span>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <small style='color: #94A3B8;'>OPTIMIZED PEAK</small><br>
                <span style='font-size: 24px; font-weight: bold;'>{peak_opt:.2f} kW</span>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <small style='color: #94A3B8;'>EST. NET SAVINGS</small><br>
                <span style='font-size: 24px; font-weight: bold; color: #4ADE80;'>${total_savings:,.2f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("### Dispatch Strategy Performance")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_res['Timestamp'], y=df_res['Baseline_Load'],
            name="Baseline Load", line=dict(color='#64748B', width=1, dash='dot')
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
        st.info("Simulation complete. Math accounts for both energy usage and demand charges.")
else:
    st.info("Configure asset specifications and click 'Execute Dispatch Strategy'.")