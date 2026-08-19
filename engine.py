import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor


class LoadForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

    def train(self, df):
        df = df.copy()
        df["h"] = df["timestamp"].dt.hour
        df["d"] = df["timestamp"].dt.dayofweek
        df["l1"] = df["load_kw"].shift(96)
        df["lw"] = df["load_kw"].shift(96 * 7)
        d = df.dropna()
        self.model.fit(d[["h", "d", "l1", "lw"]], d["load_kw"])

    def predict_24h(self, ts, history):
        if len(history) < 96 * 7:
            raise ValueError("Need at least 7 days of history before forecasting")
        l1 = history["load_kw"].tail(96).values
        lw = history["load_kw"].tail(96 * 7).iloc[:96].values
        preds = []
        for i in range(96):
            cur_ts = ts + pd.Timedelta(minutes=15 * i)
            feat = [[cur_ts.hour, cur_ts.dayofweek, l1[i], lw[i]]]
            preds.append(self.model.predict(feat)[0])
        return np.array(preds)


class RollingBrain:
    def __init__(self, cap, pwr):
        self.cap, self.max_pwr = cap, pwr
        self.eta = 0.93
        self.soc = cap * 0.5
        self.peak_so_far = 0.0
        self.deg_cost = 0.02

    def solve_full_day(self, load_f, solar_f, price_f, demand_rate, vpp_f=None):
        n = 96
        vpp_f = np.zeros(n) if vpp_f is None else vpp_f
        n_vars = 2 * n + 1
        c_idx, d_idx, p_idx = np.arange(n), np.arange(n, 2 * n), 2 * n

        obj = np.zeros(n_vars)
        obj[c_idx] = (price_f / self.eta) * 0.25
        obj[d_idx] = -((price_f + vpp_f) * self.eta) * 0.25 + self.deg_cost * 0.25
        obj[p_idx] = demand_rate
        bounds = [(0, self.max_pwr)] * (2 * n) + [(0, None)]

        A_ub, b_ub = [], []
        L = np.tril(np.ones((n, n)))

        r_max = np.zeros((n, n_vars))
        r_max[:, c_idx] = 0.25 * self.eta * L
        r_max[:, d_idx] = -0.25 / self.eta * L
        A_ub.append(r_max)
        b_ub.append(np.full(n, self.cap - self.soc))

        r_min = np.zeros((n, n_vars))
        r_min[:, c_idx] = -0.25 * self.eta * L
        r_min[:, d_idx] = 0.25 / self.eta * L
        A_ub.append(r_min)
        b_ub.append(np.full(n, self.soc))

        # Forecasted grid load must remain below the monthly peak variable.
        r_p = np.zeros((n, n_vars))
        r_p[np.arange(n), c_idx] = 1
        r_p[np.arange(n), d_idx] = -1
        r_p[np.arange(n), p_idx] = -1
        A_ub.append(r_p)
        b_ub.append(solar_f - load_f)

        r_global_p = np.zeros((1, n_vars))
        r_global_p[0, p_idx] = -1
        A_ub.append(r_global_p)
        b_ub.append(np.array([-self.peak_so_far]))

        res = linprog(obj, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub), bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"Dispatch optimization failed: {res.message}")
        return res.x[c_idx], res.x[d_idx]

    def update_state(self, c, d, actual_net):
        self.soc = max(0, min(self.cap, self.soc + c * self.eta * 0.25 - d / self.eta * 0.25))
        self.peak_so_far = max(self.peak_so_far, actual_net)


def fixed_rule_dispatch(load, solar, timestamps, cap, pwr, soc, eta=0.93):
    """Simple fixed-rule baseline: off-peak charge, 16:00-21:00 discharge."""
    charge = np.zeros(len(load))
    discharge = np.zeros(len(load))
    current_soc = float(soc)

    for i, ts in enumerate(timestamps):
        is_peak = 16 <= ts.hour < 21
        is_offpeak = not is_peak
        if is_peak and current_soc > 0:
            available = min(pwr * 0.25 / eta, current_soc)
            discharge[i] = min(pwr, available * eta / 0.25)
        elif is_offpeak and current_soc < cap:
            available = min(pwr * 0.25 * eta, cap - current_soc)
            charge[i] = min(pwr, available / (0.25 * eta))
        current_soc = max(0, min(cap, current_soc + charge[i] * eta * 0.25 - discharge[i] / eta * 0.25))

    return charge, discharge
