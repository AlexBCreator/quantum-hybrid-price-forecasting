"""Picks training settings for each network on the first 5 folds only.

Usage: python src/tune_volatility.py --train 1000   (4-year window, default; 250 = 1 year)

Grid: squash (clamp / tanh) x learning rate (0.002 / 0.01) x epochs (35 / 100),
3 seeds each, shuffled mini-batches of 16. The best setting per network goes to
results/volatility/<window>/config.json; all scores go to .../tuning.csv.
These folds are then excluded from the reported results.
"""
import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from models import HybridOrClassic  # noqa: E402
from vol_data import folds, load, standardize  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "volatility")
TUNE_FOLDS = 5
GRID = list(itertools.product(["hybrid", "classic"], ["clamp", "tanh"], [0.002, 0.01], [35, 100]))


def window_name(train):
    return f"{round(train / 250)}y"


def score(cfg):
    name, squash, lr, epochs, train = cfg
    df = load(os.path.join(ROOT, "data", "ABC.csv"))
    y = df["target"].values
    errs, base = [], []
    for k, (tr, te) in enumerate(list(folds(len(df), train=train))[:TUNE_FOLDS]):
        Xtr, ytr, ym, ys = standardize(df, tr, tr)
        Xte, _, _, _ = standardize(df, tr, te)
        for s in range(3):
            m = HybridOrClassic(6, use_quantum=(name == "hybrid"), seed=900 + 10 * k + s, squash=squash)
            m.fit(Xtr, ytr, epochs=epochs, lr=lr, batch_size=16, shuffle=True)
            errs.append(np.mean((y[te] - (m.predict(Xte) * ys + ym)) ** 2))
            base.append(np.mean((y[te] - ym) ** 2))
    return {"model": name, "squash": squash, "lr": lr, "epochs": epochs,
            "r2_oos": 1 - np.mean(errs) / np.mean(base)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1000)
    train = ap.parse_args().train
    out = os.path.join(OUT, window_name(train))
    with Pool(2) as pool:
        rows = pool.map(score, [g + (train,) for g in GRID])
    tab = pd.DataFrame(rows).sort_values(["model", "r2_oos"], ascending=[True, False])
    os.makedirs(out, exist_ok=True)
    tab.to_csv(os.path.join(out, "tuning.csv"), index=False)
    best = {n: tab[tab.model == n].iloc[0][["squash", "lr", "epochs"]].to_dict() for n in ("hybrid", "classic")}
    best = {n: {"squash": v["squash"], "lr": float(v["lr"]), "epochs": int(v["epochs"])} for n, v in best.items()}
    json.dump(best, open(os.path.join(out, "config.json"), "w"), indent=2)
    print(tab.to_string())
    print(best)


if __name__ == "__main__":
    main()
