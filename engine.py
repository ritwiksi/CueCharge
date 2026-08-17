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
        # THE FOG OF WAR: Brain guesses today's load based on yesterday
        predicted_load = yesterday_load 
        
        n_vars = 2 * n + 1 
        c_idx, d_idx, p_idx = np.arange(0, n), np.arange(n, 2 * n), 2 * n

        obj = np.zeros(n_vars)
        obj[c_idx], obj[d_idx], obj[p_idx] = price_f, -price_f, demand_rate

        bounds = [(0, self.max_pwr)] * (2 * n) + [(self.peak_so_far, None)]

        A_ub, b_ub = [], []
        L = np.tril(np.ones((n, n)))
        
        # SoC tracking
        row_max = np.zeros((n, n_vars))
        row_max[:, c_idx], row_max[:, d_idx] = self.eta * L, -(1/self.eta) * L
        A_ub.append(row_max); b_ub.append(np.full(n, self.cap - self.soc))

        row_min = np.zeros((n, n_vars))
        row_min[:, c_idx], row_min[:, d_idx] = -self.eta * L, (1/self.eta) * L
        A_ub.append(row_min); b_ub.append(np.full(n, self.soc))

        # Peak coupling: Grid Draw <= P
        row_p = np.zeros((n, n_vars))
        row_p[np.arange(n), c_idx], row_p[np.arange(n), d_idx], row_p[np.arange(n), p_idx] = 1, -1, -1
        A_ub.append(row_p)
        b_ub.append(solar_forecast - predicted_load)

        res = linprog(obj, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub), bounds=bounds, method='highs')
        
        if res.success:
            return res.x[0], res.x[n]
        return 0.0, 0.0

    def update_state(self, c, d, actual_net):
        self.soc += (c * self.eta) - (d / self.eta)
        self.soc = max(0, min(self.cap, self.soc))
        self.peak_so_far = max(self.peak_so_far, actual_net)