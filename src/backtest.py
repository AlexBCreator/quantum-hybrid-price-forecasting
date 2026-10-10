"""Directional accuracy, statistical significance and a simple long/short backtest."""
import numpy as np
from scipy import stats

TRADING_DAYS = 252


def directional_accuracy(signal, returns):
    """Share of days where the position had the same sign as the return (zero-return days excluded)."""
    nz = returns != 0
    hits = (np.sign(signal[nz]) == np.sign(returns[nz]))
    p_value = stats.binomtest(int(hits.sum()), int(nz.sum()), 0.5, alternative="greater").pvalue
    return hits.mean(), int(nz.sum()), p_value


def strategy_returns(signal, returns, cost_bps=0.0):
    """Daily log returns of holding `signal` (+1 long / -1 short). Cost per unit of turnover."""
    turnover = np.abs(np.diff(np.concatenate([[0.0], signal])))
    return signal * returns - turnover * cost_bps / 1e4


def summary(daily):
    ann_ret = np.exp(daily.mean() * TRADING_DAYS) - 1
    ann_vol = daily.std() * np.sqrt(TRADING_DAYS)
    sharpe = daily.mean() / daily.std() * np.sqrt(TRADING_DAYS) if daily.std() > 0 else 0.0
    equity = np.cumsum(daily)
    max_dd = 1 - np.exp((equity - np.maximum.accumulate(np.maximum(equity, 0))).min())
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd}
