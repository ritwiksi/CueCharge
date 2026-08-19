import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor

class LoadForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    def train(self, df):
        df = df.copy()
        df['h'], df['d'] = df['timestamp'].dt.hour, df['timestamp'].dt.dayofweek
        df['l1'], df['lw'] = df['load_kw'].shift(96), df['load_kw'].shift(96*7)
        d = df.dropna()
        self.model.fit(d[['h', 'd', 'l1', 'lw']], d['load_kw'])
    def predict_24h(self, ts, history):
        preds = []
        l1 = history['load_kw'].tail(96).values
        lw = history['load_kw'].tail(96*7).iloc[:96].values
        for i in range(96):
            cur_ts = ts + pd.Timedelta(minutes=15*i)
            feat = [[cur_ts.hour, cur_ts.dayofweek, l1[i % 96], lw[i % 96]]]
            preds.append(self.model.predict(feat)[0])
        return np.array(preds)

class RollingBrain:
    def __init__(self, cap, pwr):
        self.cap, self.max_pwr, self.eta, self.soc = cap, pwr, 0.93, cap*0.5
        self.peak_so_far, self.deg_cost = 0.0, 0.02

    def solve_full_day(self, load_f, solar_f, price_f, demand_rate, vpp_f):
        """
        Pure Dollar Optimization.
        Minimizes: (Energy Cost) - (VPP Revenue) + (Demand Charge Cost)
        """
        n = 96
        n_vars = 2 * n + 1 # [Charge(96), Discharge(96), Peak_Variable(1)]
        c_idx, d_idx, p_idx = np.arange(0, n), np.arange(n, 2 * n), 2 * n
        
        obj = np.zeros(n_vars)
        
        # 1. Cost of Charging: (Price / Efficiency) * 0.25h
        obj[c_idx] = (price_f / self.eta) * 0.25
        
        # 2. Benefit of Discharging: -(Price + VPP_Signal) * Efficiency * 0.25h
        # Plus the degradation cost ($0.02) to prevent unnecessary cycling
        obj[d_idx] = -((price_f + vpp_f) * self.eta) * 0.25 + (self.deg_cost * 0.25)
        
        # 3. Cost of the Peak: The raw PG&E Demand Rate ($/kW)
        obj[p_idx] = demand_rate 

        # BOUNDS: 
        # Charge/Discharge: [0, max_pwr]
        # Peak: [0, None] -- Let the solver find the minimum possible peak
        bounds = [(0, self.max_pwr)] * (2 * n) + [(0, None)]
        
        A_ub, b_ub = [], []
        L = np.tril(np.ones((n, n)))
        
        # --- PHYSICAL CONSTRAINTS ---
        
        # 1. SoC Max: Initial SoC + Sum(Charge) - Sum(Discharge) <= Cap
        r_max = np.zeros((n, n_vars))
        r_max[:, c_idx] = 0.25 * self.eta * L
        r_max[:, d_idx] = -0.25 * (1/self.eta) * L
        A_ub.append(r_max)
        b_ub.append(np.full(n, self.cap - self.soc))
        
        # 2. SoC Min: Initial SoC + Sum(Charge) - Sum(Discharge) >= 0
        r_min = np.zeros((n, n_vars))
        r_min[:, c_idx] = -0.25 * self.eta * L
        r_min[:, d_idx] = 0.25 * (1/self.eta) * L
        A_ub.append(r_min)
        b_ub.append(np.full(n, self.soc))

        # 3. Peak Coupling: (Load - Solar + Charge - Discharge) <= Peak_Variable
        # This is where the AI 'shaves' the peak.
        r_p = np.zeros((n, n_vars))
        r_p[np.arange(n), c_idx] = 1.0
        r_p[np.arange(n), d_idx] = -1.0
        r_p[np.arange(n), p_idx] = -1.0
        A_ub.append(r_p)
        b_ub.append(solar_f - load_f)
        
        # 4. Global Peak Constraint: Peak_Variable >= peak_so_far
        # Ensures we don't 'forget' a peak we already hit earlier in the month
        r_global_p = np.zeros((1, n_vars))
        r_global_p[0, p_idx] = -1.0
        A_ub.append(r_global_p)
        b_ub.append(np.array([-self.peak_so_far]))

        # SOLVE
        res = linprog(obj, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub), bounds=bounds, method='highs')
        
        if res.success:
            return res.x[c_idx], res.x[d_idx]
        else:
            return np.zeros(n), np.zeros(n)

    def update_state(self, c, d, actual_net):
        self.soc = max(0, min(self.cap, self.soc + (c * self.eta * 0.25) - (d / self.eta * 0.25)))
        self.peak_so_far = max(self.peak_so_far, actual_net)