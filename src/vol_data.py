"""Volatility features, target and walk-forward folds.

Daily volatility proxy: Parkinson (1980) range estimator
    pk_t = ln(High_t / Low_t)^2 / (4 ln 2)
which is several times more precise than squared close-to-close returns.

At the close of day t the model sees six features (all known at that moment):
    pk_1, pk_5, pk_22   log of Parkinson variance: last day, last week, last month (HAR structure)
    r2_5                log of mean squared close-to-close return over the last week
    ret_1               today's return (falling prices tend to raise volatility: leverage effect)
    volu                log volume today vs its 22-day mean
Target: log of mean Parkinson variance over the next 5 trading days (t+1 ... t+5).
"""
import numpy as np
import pandas as pd

FEATURES = ["pk_1", "pk_5", "pk_22", "r2_5", "ret_1", "volu"]
HORIZON = 5


def load(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    pk = np.log(df["High"] / df["Low"]) ** 2 / (4 * np.log(2))
    pk = pk.where(pk > 0, pk[pk > 0].min())          # 2 days with High == Low in 1996
    ret = np.log(df["Close"] / df["Close"].shift(1))
    out = pd.DataFrame({"date": df["Date"], "ret": ret, "pk": pk})
    out["pk_1"] = np.log(pk)
    out["pk_5"] = np.log(pk.rolling(5).mean())
    out["pk_22"] = np.log(pk.rolling(22).mean())
    out["r2_5"] = np.log((ret ** 2).rolling(5).mean().clip(lower=1e-10))
    out["ret_1"] = ret
    lv = np.log(df["Volume"])
    out["volu"] = lv - lv.rolling(22).mean()
    # future: mean pk over t+1..t+H ; next-day return for the trading application
    out["target"] = np.log(pk[::-1].rolling(HORIZON).mean()[::-1].shift(-1))
    out["ret_next"] = ret.shift(-1)
    return out.dropna().reset_index(drop=True)


def folds(n, train=1000, test=250, step=250, embargo=HORIZON):
    """Train on `train` days (default 4 years), forecast the next year (`test` days); the last `embargo` training days are
    dropped because their 5-day targets overlap the test period."""
    start = 0
    while start + train + test <= n:
        yield (np.arange(start, start + train - embargo),
               np.arange(start + train, start + train + test))
        start += step


def standardize(df, train_idx, idx):
    tr = df.iloc[train_idx]
    fm = np.array(tr[FEATURES].mean(), dtype=float)
    fs = np.array(tr[FEATURES].std(), dtype=float)
    fs[fs == 0] = 1.0
    ym, ys = float(tr["target"].mean()), float(tr["target"].std())
    X = (df[FEATURES].values[idx] - fm) / fs
    y = (df["target"].values[idx] - ym) / ys
    return X, y, ym, ys
