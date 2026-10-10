"""Builds the figures in results/figures from results/direction/predictions.csv."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from backtest import directional_accuracy, strategy_returns  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "direction", "figures")
COST_BPS = 5.0

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C = {"hybrid": "#2a78d6", "classic": "#eb6834", "logistic": "#1baf7a", "buy_and_hold": "#8a8984"}
LABEL = {"hybrid": "Квантовый гибрид", "classic": "Классический близнец",
         "logistic": "Логистическая регрессия", "buy_and_hold": "Buy & hold",
         "momentum": "Моментум"}

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 150,
    "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white",
})


def equity(pred):
    ret = pred["log_return"].values
    dates = pd.to_datetime(pred["date"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    series = [("buy_and_hold", "sig_buy_and_hold"), ("logistic", "sig_logistic"),
              ("classic", "sig_classic_ensemble"), ("hybrid", "sig_hybrid_ensemble")]
    for key, col in series:
        wealth = np.exp(np.cumsum(strategy_returns(pred[col].values, ret, COST_BPS)))
        ax.plot(dates, wealth, color=C[key], lw=2.2 if key == "hybrid" else 1.6)
        ax.annotate(f"{LABEL[key]}  ×{wealth[-1]:.2f}", (dates.iloc[-1], wealth[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9.5)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1, 2, 5, 10, 20], ["0,25", "0,5", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_ylabel("Капитал, вложено 1 (лог. шкала)")
    ax.set_title("Walk-forward 1996–2020: long/short по сигналам моделей, комиссия 5 б.п.",
                 loc="left", color=INK, fontsize=12.5)
    ax.margins(x=0.01)
    fig.savefig(os.path.join(OUT, "equity_curves.png"))
    plt.close(fig)


def accuracy(pred, seeds):
    ret = pred["log_return"].values
    rows = []
    for key in ("buy_and_hold", "momentum", "logistic"):
        acc, n, _ = directional_accuracy(pred[f"sig_{key}"].values, ret)
        rows.append((key, acc, acc, acc, n))
    for key in ("classic", "hybrid"):
        accs = [directional_accuracy(pred[f"sig_{key}_s{s}"].values, ret)[0] for s in seeds]
        rows.append((key, np.mean(accs), min(accs), max(accs), directional_accuracy(
            pred[f"sig_{key}_s0"].values, ret)[1]))
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for i, (key, m, lo, hi, n) in enumerate(rows):
        half = 1.96 * np.sqrt(0.25 / n)  # 95% band of a coin flip on n days
        ax.plot([lo * 100, hi * 100], [i, i], color=C.get(key, INK2), lw=6, alpha=0.35,
                solid_capstyle="round")
        ax.plot(m * 100, i, "o", ms=8, color=C.get(key, INK2), mec="white", mew=1.5, zorder=3)
        ax.annotate(f"{m * 100:.1f}%", (hi * 100, i), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9.5)
    ax.axvspan(50 - half * 100, 50 + half * 100, color="#d9d8d3", alpha=0.6, lw=0, zorder=0)
    ax.axvline(50, color=INK2, lw=1)
    ax.axvline(rows[0][1] * 100, color=C["buy_and_hold"], lw=1, ls="--")
    ax.text(50, -0.5, "монетка ± 95%", ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_yticks(range(len(rows)), [LABEL[r[0]] for r in rows])
    ax.set_xlabel("Точность направления, % дней (полоса — разброс по 10 seed'ам)")
    ax.set_title("Ни одна модель не точнее, чем «всегда покупать»",
                 loc="left", color=INK, fontsize=12.5)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    fig.savefig(os.path.join(OUT, "accuracy.png"))
    plt.close(fig)


def folds(pred, seeds):
    """Accuracy of every single run (seed x 50-day fold): what one notebook-style test can show."""
    ret = pred["log_return"].values
    acc = []
    for k in sorted(pred["fold"].unique()):
        m = pred["fold"].values == k
        for s in seeds:
            acc.append(directional_accuracy(pred.loc[m, f"sig_hybrid_s{s}"].values, ret[m])[0] * 100)
    acc = np.array(acc)
    share = (acc >= 57.4).mean() * 100
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bins = np.arange(np.floor(acc.min() / 2) * 2, acc.max() + 2, 2)
    ax.hist(acc, bins=bins, color=C["hybrid"], alpha=0.85, edgecolor="white", linewidth=1.5)
    ax.axvline(50, color=INK2, lw=1)
    ax.axvline(57.4, color=INK, lw=1, ls="--")
    ax.text(57.9, ax.get_ylim()[1] * 0.92, f"≥ 57,4% (результат ноутбука):\n{share:.0f}% всех запусков",
            color=INK, fontsize=9.5, va="top")
    ax.set_xlabel("Точность гибрида на одном 50-дневном тесте, %")
    ax.set_ylabel("Число запусков")
    ax.set_title(f"{len(acc)} запусков (121 отрезок × {len(seeds)} seed'ов): 57% на 50 днях — обычный шум",
                 loc="left", color=INK, fontsize=12.5)
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(OUT, "fold_accuracy.png"))
    plt.close(fig)
    return share


def main():
    os.makedirs(OUT, exist_ok=True)
    pred = pd.read_csv(os.path.join(ROOT, "results", "direction", "predictions.csv"))
    seeds = sorted(int(c.split("_s")[-1]) for c in pred.columns if c.startswith("sig_hybrid_s"))
    equity(pred)
    accuracy(pred, seeds)
    print("share of runs >= 57.4%:", round(folds(pred, seeds), 1))


if __name__ == "__main__":
    main()
