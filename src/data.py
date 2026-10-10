"""Data preparation and walk-forward folds.

Features and target are the same as in notebooks/02_regression_hybrid_vs_classic.ipynb:
  features: daily % change of Open and of Volume over the last `window` days
  target:   next day's log return of Close

Walk-forward: the notebook's setup (train on 200 days, test on the next 50) is
repeated across the whole history, moving forward 50 days at a time. Every
normalisation statistic is computed on the training block only.
"""
import numpy as np
import pandas as pd

FEATURES = ["Open_pct", "Volume_pct"]


def load(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Open_pct"] = df["Open"].pct_change()
    df["Volume_pct"] = df["Volume"].pct_change()
    return df.dropna().reset_index(drop=True)


def folds(n_rows, window=3, train=200, test=50, step=50):
    """Yields (train_days, test_days) as index arrays of *target* days."""
    start = window
    while start + train + test <= n_rows:
        yield (np.arange(start, start + train),
               np.arange(start + train, start + train + test))
        start += step


def make_xy(df, days, window, feat_mean, feat_std, ret_mean, ret_std):
    """Window of features from the `window` days before each target day."""
    feats = (df[FEATURES].values - feat_mean) / feat_std
    X = np.stack([feats[d - window:d].ravel() for d in days])   # time-major, like torch.flatten
    y = (df["log_returns"].values[days] - ret_mean) / ret_std
    return X, y


def fold_data(df, train_days, test_days, window):
    rows = df.iloc[train_days]
    fm = np.array(rows[FEATURES].mean().values, dtype=float)
    fs = np.array(rows[FEATURES].std().values, dtype=float)
    fs[fs == 0] = 1.0
    rm, rs = rows["log_returns"].mean(), rows["log_returns"].std() or 1.0
    Xtr, ytr = make_xy(df, train_days, window, fm, fs, rm, rs)
    Xte, yte = make_xy(df, test_days, window, fm, fs, rm, rs)
    return Xtr, ytr, Xte, yte, rm, rs
