"""Walk-forward volatility forecasting: hybrid quantum model vs classical twin vs standard models.

Protocol
  * Train on one year (--train 250) or four years (--train 1000), minus a 5-day embargo,
    then forecast the next year; repeat year after year.
  * The first 5 folds are used ONLY to pick training settings for each network
    (src/tune_volatility.py). Results are reported on a common period for both windows:
    from the first test date after both tuning periods (spring 2004) to March 2020.
  * 10 seeds per network; the "ensemble" averages the 10 forecasts.
  * Main score: mean squared error of the log forecast, and the gain over the naive
    forecast ("next week = last week"), which does not depend on the training window.

Usage:  python src/tune_volatility.py --train 250
        python src/run_volatility.py --train 250 --seeds 10 --workers 2
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from backtest import summary  # noqa: E402
from models import HybridOrClassic  # noqa: E402
from vol_baselines import ewma_variance, forecasts  # noqa: E402
from vol_data import folds, load, standardize  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "volatility")
TUNE_FOLDS = 5
COST_BPS = 5.0


def report_start(df):
    """First test day after the tuning folds of the longest (4-year) window."""
    return list(folds(len(df), train=1000))[TUNE_FOLDS][1][0]


def run_seed(args):
    seed, cfg, train = args
    df = load(os.path.join(ROOT, "data", "ABC.csv"))
    res = {"hybrid": ([], []), "classic": ([], [])}
    t0 = time.time()
    for k, (tr, te) in enumerate(folds(len(df), train=train)):
        Xtr, ytr, ym, ys = standardize(df, tr, tr)
        Xte, _, _, _ = standardize(df, tr, te)
        for name, q in (("hybrid", True), ("classic", False)):
            c = cfg[name]
            m = HybridOrClassic(6, use_quantum=q, seed=seed * 1000 + k, squash=c["squash"])
            m.fit(Xtr, ytr, epochs=c["epochs"], lr=c["lr"], batch_size=16, shuffle=True)
            res[name][0].append(m.predict(Xtr) * ys + ym)
            res[name][1].append(m.predict(Xte) * ys + ym)
        if k % 5 == 0:
            print(f"seed {seed} fold {k} {time.time() - t0:.0f}s", flush=True)
    return seed, res


def diebold_mariano(loss_a, loss_b, lag=5):
    """DM test with Newey-West variance (targets overlap over 5 days). Negative stat: a is better."""
    d = loss_a - loss_b
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for L in range(1, lag + 1):
        var += 2 * (1 - L / (lag + 1)) * (dc[L:] @ dc[:-L]) / n
    stat = d.mean() / np.sqrt(var / n)
    from scipy.stats import norm
    return stat, 2 * (1 - norm.cdf(abs(stat)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--train", type=int, default=1000)
    args = ap.parse_args()
    out_dir = os.path.join(OUT, f"{round(args.train / 250)}y")
    config = json.load(open(os.path.join(out_dir, "config.json")))

    df = load(os.path.join(ROOT, "data", "ABC.csv"))
    y, ret_next, ret = df["target"].values, df["ret_next"].values, df["ret"].values
    F = list(folds(len(df), train=args.train))
    ewma_all = ewma_variance(df["pk"].values)

    names = ["naive", "ewma", "garch", "har", "linear"]
    fc = {n: ([], []) for n in names}
    for tr, te in F:
        f = forecasts(df, tr, te, ewma_all)
        for n in names:
            fc[n][0].append(f[n][0])
            fc[n][1].append(f[n][1])

    seeds = list(range(args.seeds))
    with Pool(args.workers) as pool:
        out = pool.map(run_seed, [(s, config, args.train) for s in seeds])
    for s, res in sorted(out):
        for name in ("hybrid", "classic"):
            fc[f"{name}_s{s}"] = res[name]
    for name in ("hybrid", "classic"):
        fc[f"{name}_ensemble"] = tuple(
            [np.mean([fc[f"{name}_s{s}"][j][k] for s in seeds], axis=0) for k in range(len(F))]
            for j in (0, 1))

    # long table of test forecasts
    rows = []
    for k, (tr, te) in enumerate(F):
        block = pd.DataFrame({"date": df["date"].values[te], "fold": k, "target": y[te],
                              "ret": ret[te], "ret_next": ret_next[te],
                              "train_ret_std": ret[tr].std(),
                              "pk_to_cc": np.mean(ret[tr] ** 2) / np.mean(df["pk"].values[tr])})
        for n, (ftr, fte) in fc.items():
            block[f"f_{n}"] = fte[k]
            block[f"c_{n}"] = np.mean(np.exp(y[tr])) / np.mean(np.exp(ftr[k]))
        rows.append(block)
        block["idx"] = te
    pred = pd.concat(rows, ignore_index=True)
    pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # metrics on the common reported period
    test = pred["idx"].values >= report_start(df)
    yt = pred["target"].values[test]
    base_mse = np.mean((yt - pred["f_naive"].values[test]) ** 2)
    metric_rows = []
    losses = {}
    for n in fc:
        p = pred[f"f_{n}"].values[test]
        se = (yt - p) ** 2
        losses[n] = se
        c = pred[f"c_{n}"].values[test]
        ratio = np.exp(yt) / (np.exp(p) * c)
        metric_rows.append({"model": n, "gain_vs_naive": 1 - se.mean() / base_mse, "mse_log": se.mean(),
                            "qlike": np.mean(ratio - np.log(ratio) - 1)})
    metrics = pd.DataFrame(metric_rows)

    tests = []
    for a, b in [("hybrid_ensemble", "classic_ensemble"), ("hybrid_ensemble", "har"),
                 ("classic_ensemble", "har"), ("ewma", "har"), ("hybrid_ensemble", "ewma")]:
        stat, p = diebold_mariano(losses[a], losses[b])
        tests.append({"a": a, "b": b, "dm_stat": stat, "p_value": p})
    # per-seed paired comparison: hybrid vs classic with the same seed
    wins = sum(losses[f"hybrid_s{s}"].mean() < losses[f"classic_s{s}"].mean() for s in seeds)
    tests.append({"a": "hybrid_s*", "b": "classic_s*", "dm_stat": np.nan,
                  "p_value": np.nan, "note": f"hybrid better in {wins}/{len(seeds)} seeds"})

    # application: volatility targeting at the training-period volatility, leverage cap 2
    apps = []
    sub = pred[test]
    for n in ["buy_and_hold", "naive", "ewma", "har", "classic_ensemble", "hybrid_ensemble"]:
        if n == "buy_and_hold":
            w = np.ones(len(sub))
        else:
            var_cc = np.exp(sub[f"f_{n}"].values) * sub[f"c_{n}"].values * sub["pk_to_cc"].values
            w = np.minimum(2.0, sub["train_ret_std"].values / np.sqrt(var_cc))
        turnover = np.abs(np.diff(np.concatenate([[w[0]], w])))
        daily = w * sub["ret_next"].values - turnover * COST_BPS / 1e4
        s = summary(daily)
        s.update({"strategy": n, "avg_leverage": w.mean()})
        apps.append(s)

    metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(out_dir, "dm_tests.csv"), index=False)
    pd.DataFrame(apps).to_csv(os.path.join(out_dir, "vol_targeting.csv"), index=False)
    print("reported period:", df["date"].iloc[report_start(df)].date(), "-", pred["date"].iloc[-1])
    print(metrics.to_string())
    print(pd.DataFrame(tests).to_string())
    print(pd.DataFrame(apps).to_string())


if __name__ == "__main__":
    main()
