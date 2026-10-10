"""Walk-forward comparison: hybrid quantum model vs classical twin vs simple baselines.

Usage:  python src/run_direction.py --seeds 10 --workers 2
Writes results/direction/predictions.csv, metrics.csv, folds.csv
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from backtest import directional_accuracy, strategy_returns, summary  # noqa: E402
from data import fold_data, folds, load  # noqa: E402
from models import HybridOrClassic  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = dict(window=3, train=200, test=50, step=50, epochs=35, batch_size=16, lr=0.002)
COST_BPS = 5.0


def run_seed(seed):
    df = load(os.path.join(ROOT, "data", "ABC.csv"))
    out = {"hybrid": [], "classic": []}
    t0 = time.time()
    for k, (tr, te) in enumerate(folds(len(df), CFG["window"], CFG["train"], CFG["test"], CFG["step"])):
        Xtr, ytr, Xte, _, rm, rs = fold_data(df, tr, te, CFG["window"])
        for name, q in (("hybrid", True), ("classic", False)):
            m = HybridOrClassic(Xtr.shape[1], use_quantum=q, seed=seed * 1000 + k)
            m.fit(Xtr, ytr, CFG["epochs"], CFG["batch_size"], CFG["lr"])
            out[name].append(m.predict(Xte) * rs + rm)
        if k % 20 == 0:
            print(f"seed {seed} fold {k} {time.time() - t0:.0f}s", flush=True)
    return seed, {n: np.concatenate(v) for n, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    df = load(os.path.join(ROOT, "data", "ABC.csv"))
    fold_list = list(folds(len(df), CFG["window"], CFG["train"], CFG["test"], CFG["step"]))
    test_days = np.concatenate([te for _, te in fold_list])
    fold_id = np.concatenate([np.full(len(te), k) for k, (_, te) in enumerate(fold_list)])
    ret = df["log_returns"].values[test_days]

    pred = pd.DataFrame({"date": df["Date"].values[test_days], "fold": fold_id, "log_return": ret})
    # Baselines
    pred["sig_buy_and_hold"] = 1.0
    prev = df["log_returns"].values[test_days - 1]
    pred["sig_momentum"] = np.where(prev >= 0, 1.0, -1.0)
    logit = []
    for tr, te in fold_list:
        Xtr, ytr, Xte, _, rm, rs = fold_data(df, tr, te, CFG["window"])
        clf = LogisticRegression().fit(Xtr, (ytr * rs + rm) > 0)
        logit.append(np.where(clf.predict_proba(Xte)[:, 1] > 0.5, 1.0, -1.0))
    pred["sig_logistic"] = np.concatenate(logit)

    seeds = list(range(args.seeds))
    with Pool(args.workers) as pool:
        results = pool.map(run_seed, seeds)
    for seed, res in sorted(results):
        for name, p in res.items():
            pred[f"pred_{name}_s{seed}"] = p
            pred[f"sig_{name}_s{seed}"] = np.where(p > 0, 1.0, -1.0)
    for name in ("hybrid", "classic"):
        ens = pred[[f"pred_{name}_s{s}" for s in seeds]].mean(axis=1).values
        pred[f"sig_{name}_ensemble"] = np.where(ens > 0, 1.0, -1.0)

    os.makedirs(os.path.join(ROOT, "results", "direction"), exist_ok=True)
    pred.to_csv(os.path.join(ROOT, "results", "direction", "predictions.csv"), index=False)

    rows = []
    for col in [c for c in pred.columns if c.startswith("sig_")]:
        sig = pred[col].values
        acc, n, p = directional_accuracy(sig, ret)
        gross = summary(strategy_returns(sig, ret, 0.0))
        net = summary(strategy_returns(sig, ret, COST_BPS))
        rows.append({"model": col[4:], "accuracy": acc, "n_days": n, "p_value": p,
                     "sharpe_gross": gross["sharpe"], "sharpe_net": net["sharpe"],
                     "ann_return_net": net["ann_return"], "max_drawdown_net": net["max_drawdown"]})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(ROOT, "results", "direction", "metrics.csv"), index=False)

    # Per-fold accuracy (to show how noisy a single 50-day test is)
    fold_rows = []
    for k in range(len(fold_list)):
        m = pred["fold"].values == k
        r = {"fold": k, "start": pred["date"].values[m][0]}
        for name in ("hybrid", "classic"):
            accs = [directional_accuracy(pred.loc[m, f"sig_{name}_s{s}"].values, ret[m])[0] for s in seeds]
            r[f"{name}_acc_mean"] = np.mean(accs)
        fold_rows.append(r)
    pd.DataFrame(fold_rows).to_csv(os.path.join(ROOT, "results", "direction", "folds.csv"), index=False)
    print(metrics.to_string())


if __name__ == "__main__":
    main()
