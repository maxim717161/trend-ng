"""Подбор размера окна трендовой стратегии на NGM (минутки, контракты не склеиваются).

Схема для каждого завершённого контракта отдельно:
  [экспирация-2м, экспирация-1м)  - прогрев: индикаторы считаются, позиции нет
  [экспирация-1м, экспирация]     - торговля; на последней минуте позиция закрывается

Сигнал по закрытию бара t исполняется по open бара t+1. Комиссия 0.05% от номинала
за каждую сторону (изменение позиции на 1.0 = одна сторона).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "minute"
FEE = 0.0005
WINDOWS = [5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 240, 300, 360, 480, 600, 720, 840, 960, 1200, 1440, 1920,
           2880, 3600, 4320, 5760, 7200, 8640, 10080, 12000, 15000]
# Сигнал включается только когда накоплено достаточно баров для окна (иначе позиция 0):
# donchian/momentum - n+1 баров, ema_cross - SLOW_MULT*n баров (медленная EMA).
# Если месяца прогрева не хватает, стратегия начинает торговать позже внутри торгового месяца.
FAMILIES = ("donchian", "ema_cross", "momentum")
SLOW_MULT = 4  # slow EMA = SLOW_MULT * fast


@dataclass
class Contract:
    secid: str
    expiry: pd.Timestamp
    warm_start: pd.Timestamp
    trade_start: pd.Timestamp
    bars: pd.DataFrame  # индекс - begin; open/high/low/close/volume


def load_contracts(min_warm_days: int = 25) -> list[Contract]:
    files = sorted(DATA.glob("*_1m.parquet"))
    raw = {p.name.split("_")[0]: pd.read_parquet(p).set_index("begin") for p in files}
    last_ts = max(df.index.max() for df in raw.values())
    out = []
    for secid, df in raw.items():
        if df.index.max() >= last_ts - pd.Timedelta(days=1):
            continue  # контракт ещё торгуется
        expiry = df.index.max().normalize()
        trade_start = expiry - pd.DateOffset(months=1)
        warm_start = expiry - pd.DateOffset(months=2)
        if df.index.min() > warm_start + pd.Timedelta(days=30 - min_warm_days):
            continue  # нет полного месяца прогрева
        sl = df.loc[warm_start:]
        out.append(Contract(secid, expiry, warm_start, trade_start, sl))
    return sorted(out, key=lambda c: c.expiry)


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def signal(family: str, close: np.ndarray, n: int) -> np.ndarray:
    """Целевая позиция {-1,0,1} по закрытию каждого бара."""
    c = pd.Series(close)
    if family == "donchian":
        hi = c.rolling(n).max().shift(1)
        lo = c.rolling(n).min().shift(1)
        s = pd.Series(np.nan, index=c.index)
        s[c > hi] = 1.0
        s[c < lo] = -1.0
        return s.ffill().fillna(0.0).to_numpy()
    if family == "ema_cross":
        return np.sign(_ema(close, n) - _ema(close, SLOW_MULT * n))
    if family == "momentum":
        return np.nan_to_num(np.sign(close - c.shift(n).to_numpy()))
    raise ValueError(family)


def lookback(family: str, n: int) -> int:
    return SLOW_MULT * n if family == "ema_cross" else n + 1


def backtest(ct: Contract, family: str, n: int, fee: float = FEE) -> dict:
    b = ct.bars
    o, c = b["open"].to_numpy(), b["close"].to_numpy()
    ts = b.index
    sig = signal(family, c, n)
    in_trade = ts >= ct.trade_start
    ready = np.arange(len(c)) + 1 >= lookback(family, n)
    sig = np.where(in_trade & ready, sig, 0.0)
    # позиция, действующая в баре t: сигнал t-1 (исполнение по open t)
    pos_new = np.r_[0.0, sig[:-1]]
    pos_new[-1] = 0.0  # закрытие по open последнего бара
    pos_old = np.r_[0.0, pos_new[:-1]]
    prev_c = np.r_[c[0], c[:-1]]
    gap = o / prev_c - 1
    intra = c / o - 1
    gross = pos_old * gap + pos_new * intra
    turnover = np.abs(pos_new - pos_old)
    cost = fee * turnover
    # учёт только торговой части
    m = in_trade
    # последний бар: позиция закрыта по open, внутрибарного P&L нет
    return {
        "gross": gross[m].sum(),
        "fees": cost[m].sum(),
        "net": (gross - cost)[m].sum(),
        "trades": int(((turnover > 0) & m).sum()),
        "bars": int(m.sum()),
        "coverage": float(ready[m].mean()),
        "warm_ok": bool(ready[np.argmax(in_trade)]),
        "pnl": pd.Series((gross - cost)[m], index=ts[m]),
    }


def sweep(contracts: list[Contract], windows=WINDOWS, families=FAMILIES) -> pd.DataFrame:
    rows = []
    for ct in contracts:
        bh = ct.bars.loc[ct.trade_start:]
        buyhold = bh["close"].iloc[-1] / bh["open"].iloc[0] - 1
        for f in families:
            for n in windows:
                r = backtest(ct, f, n)
                rows.append({"secid": ct.secid, "expiry": ct.expiry, "family": f, "n": n,
                             "gross": r["gross"], "fees": r["fees"], "net": r["net"],
                             "trades": r["trades"], "bars": r["bars"], "coverage": r["coverage"],
                             "warm_ok": r["warm_ok"], "buyhold": buyhold})
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby(["family", "n"])
    s = g.agg(contracts=("net", "size"), net_mean=("net", "mean"), net_median=("net", "median"),
              net_std=("net", "std"), gross_mean=("gross", "mean"), fees_mean=("fees", "mean"),
              trades_mean=("trades", "mean"), coverage=("coverage", "mean"), win_rate=("net", lambda x: (x > 0).mean()))
    s["t_stat"] = s["net_mean"] / (s["net_std"] / np.sqrt(s["contracts"]))
    return s.reset_index()
