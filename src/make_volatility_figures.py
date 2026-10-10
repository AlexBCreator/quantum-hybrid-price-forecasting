"""Figures for the volatility experiment (results/volatility/figures)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
VOL = os.path.join(ROOT, "results", "volatility")
OUT = os.path.join(VOL, "figures")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
LABEL = {"naive": "Наивная (как прошлая неделя)", "ewma": "EWMA (RiskMetrics)", "garch": "GARCH(1,1)",
         "har": "HAR", "linear": "Линейная регрессия", "classic_ensemble": "Классический близнец",
         "hybrid_ensemble": "Квантовый гибрид"}
ORDER = ["hybrid_ensemble", "classic_ensemble", "linear", "har", "garch", "ewma"]

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 150,
    "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white",
})


def ann_vol(log_var):
    return np.sqrt(np.exp(log_var) * 252) * 100


def model_comparison():
    m1 = pd.read_csv(os.path.join(VOL, "1y", "metrics.csv")).set_index("model")
    m4 = pd.read_csv(os.path.join(VOL, "4y", "metrics.csv")).set_index("model")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(ORDER))
    h = 0.36
    for off, m, color, lab in [(-h / 2, m4, BLUE, "Обучение на 4 годах (основное)"),
                               (h / 2, m1, GRAY, "Обучение на 1 годе (проверка)")]:
        vals = [m.loc[k, "gain_vs_naive"] * 100 for k in ORDER]
        ax.barh(y + off, vals, height=h - 0.04, color=color, label=lab)
        for yi, v in zip(y + off, vals):
            ax.text(v + (0.6 if v >= 0 else -0.6), yi, f"{v:+.0f}%", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [LABEL[k] for k in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Снижение ошибки прогноза относительно наивного, %  (больше — лучше)")
    ax.set_title("Прогноз волатильности на 5 дней, 2004–2020", loc="left", color=INK, fontsize=12.5)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower left", fontsize=9.5)
    lo = min(0, min(m1["gain_vs_naive"].min(), m4["gain_vs_naive"].min()) * 100) - 6
    hi = max(m1["gain_vs_naive"].max(), m4["gain_vs_naive"].max()) * 100 + 8
    ax.set_xlim(lo, hi)
    fig.savefig(os.path.join(OUT, "model_comparison.png"))
    plt.close(fig)


def crisis_zoom(window="4y"):
    p = pd.read_csv(os.path.join(VOL, window, "predictions.csv"), parse_dates=["date"])
    z = p[(p.date >= "2007-06-01") & (p.date <= "2009-12-31")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(z.date, ann_vol(z.target), color=GRAY, lw=1.0, label="Факт (следующие 5 дней)")
    ax.plot(z.date, ann_vol(z.f_ewma), color=AQUA, lw=1.6, label="EWMA")
    ax.plot(z.date, ann_vol(z.f_hybrid_ensemble), color=BLUE, lw=2, label="Квантовый гибрид")
    ax.set_ylabel("Волатильность, % годовых")
    ax.set_title("Осень 2008: EWMA ловит всплеск волатильности, гибрид реагирует слабо",
                 loc="left", color=INK, fontsize=12.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.savefig(os.path.join(OUT, "crisis_2008.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    model_comparison()
    crisis_zoom()


if __name__ == "__main__":
    main()
