"""Standard volatility models used as baselines. All forecast the log of mean daily
Parkinson variance over the next 5 days, using only information available at day t.

  naive   last week's variance (random walk)
  ewma    RiskMetrics exponential smoothing, lambda = 0.94
  garch   GARCH(1,1) on returns, fitted by maximum likelihood on each training window
  har     HAR regression (Corsi, 2009): OLS on last day / week / month variance
  linear  OLS on all six features the neural networks see

EWMA and GARCH forecast a different scale (GARCH models close-to-close returns, which
include overnight moves), so their log forecasts get an intercept fitted on the training
window: this only shifts the level and keeps the comparison fair.
"""
import numpy as np
from scipy.optimize import minimize

from vol_data import FEATURES, HORIZON


def ewma_variance(pk, lam=0.94):
    v = np.empty_like(pk)
    v[0] = pk[:22].mean()
    for t in range(1, len(pk)):
        v[t] = lam * v[t - 1] + (1 - lam) * pk[t]
    return v


def _garch_filter(r, omega, alpha, beta, v0):
    v = np.empty(len(r) + 1)
    v[0] = v0
    for t in range(len(r)):
        v[t + 1] = omega + alpha * r[t] ** 2 + beta * v[t]
    return v  # v[t+1] = variance for day t+1 given info up to day t


def garch_fit(r):
    v0 = r.var()

    def nll(p):
        omega, alpha, beta = p
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10
        v = _garch_filter(r, omega, alpha, beta, v0)[:-1]
        return 0.5 * np.sum(np.log(v) + r ** 2 / v)

    best = None
    for a, b in [(0.05, 0.90), (0.10, 0.85), (0.03, 0.95)]:
        res = minimize(nll, [v0 * (1 - a - b), a, b], method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def garch_forecast(df, train_idx, idx):
    """5-day mean variance forecast for days idx, with parameters fitted on train_idx."""
    r_all = df["ret"].values
    mu = r_all[train_idx].mean()
    omega, alpha, beta = garch_fit(r_all[train_idx] - mu)
    start = train_idx[0]
    r = r_all[start:idx[-1] + 1] - mu
    v_next = _garch_filter(r, omega, alpha, beta, r_all[train_idx].var())[1:]  # for day t+1
    persist = alpha + beta
    long_run = omega / (1 - persist)
    k = np.arange(HORIZON)
    mult = np.mean(persist ** k)
    mean5 = long_run + mult * (v_next - long_run)
    return mean5[idx - start]


def _ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return lambda Z: np.column_stack([np.ones(len(Z)), Z]) @ coef


def forecasts(df, train_idx, test_idx, ewma_all):
    """Returns {name: (train_forecast, test_forecast)} of the log target."""
    y = df["target"].values
    out = {}
    out["naive"] = (df["pk_5"].values[train_idx], df["pk_5"].values[test_idx])
    for name, raw_tr, raw_te in [
        ("ewma", np.log(ewma_all[train_idx]), np.log(ewma_all[test_idx])),
        ("garch", np.log(garch_forecast(df, train_idx, train_idx)),
         np.log(garch_forecast(df, train_idx, test_idx))),
    ]:
        shift = np.mean(y[train_idx] - raw_tr)
        out[name] = (raw_tr + shift, raw_te + shift)
    for name, cols in [("har", ["pk_1", "pk_5", "pk_22"]), ("linear", FEATURES)]:
        f = _ols(df[cols].values[train_idx], y[train_idx])
        out[name] = (f(df[cols].values[train_idx]), f(df[cols].values[test_idx]))
    return out
