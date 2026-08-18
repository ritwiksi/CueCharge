import numpy as np
from scipy.optimize import linprog

class RollingBrain:
    def __init__(self, cap_kwh, max_kw, eta=0.93):
        self.cap = cap_kwh
        self.max_pwr = max_kw
        self.eta = eta
        self.soc = cap_kwh * 0.5
        self.peak_so_far = 0.0

    def solve_best_move(self, yesterday_load, solar_forecast, price_f, demand_rate):
        n = 24
        # 1. HEDGING: Predict load as yesterday + 10% safety margin to prevent 
        # undershooting the peak shave.
        predicted_load = yesterday_load * 1.05 
        
        n_vars = 2 * n + 1 
        c_idx, d_idx, p_idx = np.arange(0, n), np.arange(n, 2 * n), 2 * n

        # 2. EFFICIENCY-AWARE OBJECTIVE
        # We must pay for charging (price / eta) and get paid for discharging (price * eta)
        # This prevents the battery from 'cycling' unless the price spread > round-trip loss.
        obj = np.zeros(n_vars)
        obj[c_idx] = price_f / self.eta 
        obj[d_idx] = -price_f * self.eta
        
        # 3. DEMAND WEIGHTING
        # Since we solve 1 day but bill 30 days, we weight the demand charge 
        # so the LP takes it seriously but doesn't ignore energy arbitrage.
        obj[p_idx] = demand_rate 

        bounds = [(0, self.max_pwr)] * (2 * n) + [(self.peak_so_far, None)]

        A_ub, b_ub = [], []
        L = np.tril(np.ones((n, n)))
        
        # SoC tracking: Initial SoC + Sum(Charge) - Sum(Discharge)
        # 0 <= SoC <= Cap
        row_max = np.zeros((n, n_vars))
        row_max[:, c_idx], row_max[:, d_idx] = self.eta * L, -(1/self.eta) * L
        A_ub.append(row_max); b_ub.append(np.full(n, self.cap - self.soc))

        row_min = np.zeros((n, n_vars))
        row_min[:, c_idx], row_min[:, d_idx] = -self.eta * L, (1/self.eta) * L
        A_ub.append(row_min); b_ub.append(np.full(n, self.soc))

        # Peak coupling: (Load - Solar + Charge - Discharge) <= Peak_Variable
        # This allows the battery to 'shave' by increasing D or decreasing C.
        row_p = np.zeros((n, n_vars))
        row_p[np.arange(n), c_idx] = 1.0
        row_p[np.arange(n), d_idx] = -1.0
        row_p[np.arange(n), p_idx] = -1.0
        A_ub.append(row_p)
        
        # Grid Draw Limit: Solar - Load
        b_ub.append(solar_forecast - predicted_load)

        res = linprog(obj, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub), bounds=bounds, method='highs')
        
        if res.success:
            # Return first hour actions
            return res.x[0], res.x[n]
        return 0.0, 0.0

    def update_state(self, c, d, actual_net):
        # Update physical state
        self.soc += (c * self.eta) - (d / self.eta)
        self.soc = max(0, min(self.cap, self.soc))
        # Important: The 'Peak So Far' for the next step should be based 
        # on the ACTUAL grid draw observed, not the predicted one.
        self.peak_so_far = max(self.peak_so_far, actual_net)