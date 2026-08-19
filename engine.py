import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor


class LoadForecaster:
    """Load forecast using same-time yesterday/last-week features."""

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    def train(self, df):
        df = df.copy()
        df["h"] = df["timestamp"].dt.hour
        df["q"] = df["timestamp"].dt.minute // 15
        df["d"] = df["timestamp"].dt.dayofweek
        df["l1"] = df["load_kw"].shift(96)
        df["lw"] = df["load_kw"].shift(96 * 7)
        d = df.dropna()
        self.model.fit(d[["h", "q", "d", "l1", "lw"]], d["load_kw"])

    def predict_24h(self, ts, history):
        """Forecast the next 96 intervals using only observations before ts."""
        if len(history) < 96 * 7:
            raise ValueError("Need at least 7 days of history before forecasting")
        indexed = history.set_index("timestamp")["load_kw"]
        preds = []
        for i in range(96):
            cur_ts = ts + pd.Timedelta(minutes=15 * i)
            try:
                l1 = float(indexed.loc[cur_ts - pd.Timedelta(days=1)])
                lw = float(indexed.loc[cur_ts - pd.Timedelta(days=7)])
            except KeyError as exc:
                raise ValueError(f"Missing historical load needed for forecast at {cur_ts}") from exc
            feat = [[cur_ts.hour, cur_ts.minute // 15, cur_ts.dayofweek, l1, lw]]
            preds.append(self.model.predict(feat)[0])
        return np.maximum(np.asarray(preds), 0.0)


def persistence_forecast(column, ts, history, horizon=96):
    """Use the same timestamp from the previous day as a transparent baseline forecast."""
    indexed = history.set_index("timestamp")[column]
    values = []
    for i in range(horizon):
        target = ts + pd.Timedelta(minutes=15 * i) - pd.Timedelta(days=1)
        try:
            values.append(float(indexed.loc[target]))
        except KeyError as exc:
            raise ValueError(f"Missing historical {column} needed for forecast at {target}") from exc
    return np.maximum(np.asarray(values), 0.0)


class RollingBrain:
    """Forecast-informed rolling-horizon battery dispatcher."""

    def __init__(self, cap, pwr, eta_charge=0.95, eta_discharge=0.95, min_soc_frac=0.10):
        self.cap = float(cap)
        self.max_pwr = float(pwr)
        self.eta_charge = float(eta_charge)
        self.eta_discharge = float(eta_discharge)
        self.min_soc = self.cap * float(min_soc_frac)
        self.soc = self.cap * 0.5
        self.peak_so_far = 0.0
        self.deg_cost = 0.02

    def solve_horizon(self, load_f, solar_f, price_f, demand_rate, terminal_soc=None):
        """Optimize a 15-minute horizon; the caller executes only the first step."""
        n = len(load_f)
        if not (len(solar_f) == len(price_f) == n):
            raise ValueError("Forecast arrays must have equal length")

        c_idx = np.arange(n)
        d_idx = np.arange(n, 2 * n)
        p_idx = 2 * n
        n_vars = 2 * n + 1
        dt = 0.25

        obj = np.zeros(n_vars)
        obj[c_idx] = (price_f / self.eta_charge) * dt + self.deg_cost * dt
        obj[d_idx] = -(price_f * self.eta_discharge) * dt + self.deg_cost * dt
        obj[p_idx] = demand_rate
        bounds = [(0, self.max_pwr)] * (2 * n) + [(0, None)]

        A_ub, b_ub = [], []
        L = np.tril(np.ones((n, n)))

        # SOC <= capacity.
        r_max = np.zeros((n, n_vars))
        r_max[:, c_idx] = dt * self.eta_charge * L
        r_max[:, d_idx] = -dt / self.eta_discharge * L
        A_ub.append(r_max)
        b_ub.append(np.full(n, self.cap - self.soc))

        # SOC >= minimum reserve.
        r_min = np.zeros((n, n_vars))
        r_min[:, c_idx] = -dt * self.eta_charge * L
        r_min[:, d_idx] = dt / self.eta_discharge * L
        A_ub.append(r_min)
        b_ub.append(np.full(n, self.soc - self.min_soc))

        # Shared inverter limit.
        r_power = np.zeros((n, n_vars))
        r_power[np.arange(n), c_idx] = 1
        r_power[np.arange(n), d_idx] = 1
        A_ub.append(r_power)
        b_ub.append(np.full(n, self.max_pwr))

        # Forecasted grid load <= monthly peak variable.
        r_peak = np.zeros((n, n_vars))
        r_peak[np.arange(n), c_idx] = 1
        r_peak[np.arange(n), d_idx] = -1
        r_peak[np.arange(n), p_idx] = -1
        A_ub.append(r_peak)
        b_ub.append(np.asarray(solar_f) - np.asarray(load_f))

        # Peak variable cannot fall below the observed billing-period peak.
        r_global_peak = np.zeros((1, n_vars))
        r_global_peak[0, p_idx] = -1
        A_ub.append(r_global_peak)
        b_ub.append(np.array([-self.peak_so_far]))

        # Preserve a useful SOC at the end of the look-ahead horizon.
        if terminal_soc is None:
            terminal_soc = self.cap * 0.50
        terminal_soc = max(self.min_soc, min(self.cap, float(terminal_soc)))
        r_terminal = np.zeros((1, n_vars))
        r_terminal[0, c_idx] = -dt * self.eta_charge
        r_terminal[0, d_idx] = dt / self.eta_discharge
        A_ub.append(r_terminal)
        b_ub.append(np.array([self.soc - terminal_soc]))

        res = linprog(
            obj,
            A_ub=np.vstack(A_ub),
            b_ub=np.concatenate(b_ub),
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"Dispatch optimization failed: {res.message}")
        return res.x[c_idx], res.x[d_idx]

    def solve_full_day(self, load_f, solar_f, price_f, demand_rate):
        return self.solve_horizon(load_f, solar_f, price_f, demand_rate)

    def update_state(self, c, d, actual_net):
        self.soc = max(
            self.min_soc,
            min(self.cap, self.soc + c * self.eta_charge * 0.25 - d / self.eta_discharge * 0.25),
        )
        self.peak_so_far = max(self.peak_so_far, actual_net)


def fixed_rule_dispatch(load, solar, timestamps, cap, pwr, soc, eta=0.93):
    """Simple fixed-rule baseline: charge off-peak, discharge 16:00-21:00."""
    charge = np.zeros(len(load))
    discharge = np.zeros(len(load))
    current_soc = float(soc)
    min_soc = cap * 0.10

    for i, ts in enumerate(timestamps):
        is_peak = 16 <= ts.hour < 21
        if is_peak and current_soc > min_soc:
            available = min(pwr * 0.25 / eta, current_soc - min_soc)
            discharge[i] = min(pwr, available * eta / 0.25)
        elif not is_peak and current_soc < cap:
            available = min(pwr * 0.25 * eta, cap - current_soc)
            charge[i] = min(pwr, available / (0.25 * eta))
        current_soc = max(
            min_soc,
            min(cap, current_soc + charge[i] * eta * 0.25 - discharge[i] / eta * 0.25),
        )

    return charge, discharge
