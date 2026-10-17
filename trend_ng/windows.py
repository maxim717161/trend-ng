"""Поиск окон для исследования трендовой стратегии на NGM.

Окно исследования = календарное окно длиной WINDOW_DAYS. Перед каждым окном
берётся месяц прогрева (WARMUP_DAYS): индикаторы считаются, позиция = 0.

Метрики трендовости считаются ТОЛЬКО по ценам окна и не зависят от PnL:
  ER   - efficiency ratio Кауфмана на 4ч барах (|чистое движение| / путь)
  R2   - R^2 регрессии лог-цены на время
  move - чистое движение за окно
Параллельно прогоняется набор простых трендовых стратегий (комиссия 0.05% за
сторону от номинала) - как проверка, что выбранные окна действительно
торгуются трендовой логикой, а не просто выглядят красиво.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
FEE = 0.0005          # 0.05% за сторону
WINDOW_DAYS = 30
WARMUP_DAYS = 30
STEP_DAYS = 7


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "ngm_continuous_60m.parquet")
    return df.set_index("begin")


# ---------- стратегии: позиция, определённая по закрытию бара t, действует на баре t+1

def donchian(logp: pd.Series, n: int) -> pd.Series:
    hi = logp.rolling(n).max().shift(1)
    lo = logp.rolling(n).min().shift(1)
    pos = pd.Series(np.nan, index=logp.index)
    pos[logp > hi] = 1.0
    pos[logp < lo] = -1.0
    return pos.ffill().fillna(0.0)


def ema_cross(logp: pd.Series, fast: int, slow: int) -> pd.Series:
    f = logp.ewm(span=fast, adjust=False).mean()
    s = logp.ewm(span=slow, adjust=False).mean()
    return np.sign(f - s).astype(float)


def momentum(logp: pd.Series, n: int) -> pd.Series:
    return np.sign(logp - logp.shift(n)).fillna(0.0)


STRATEGIES = {
    "donchian_48": lambda p: donchian(p, 48),
    "donchian_120": lambda p: donchian(p, 120),
    "donchian_240": lambda p: donchian(p, 240),
    "ema_12_48": lambda p: ema_cross(p, 12, 48),
    "ema_24_96": lambda p: ema_cross(p, 24, 96),
    "ema_48_192": lambda p: ema_cross(p, 48, 192),
    "mom_96": lambda p: momentum(p, 96),
    "mom_240": lambda p: momentum(p, 240),
}


def run_strategy(df: pd.DataFrame, name: str, start: pd.Timestamp, end: pd.Timestamp,
                 warmup: int = WARMUP_DAYS, fee: float = FEE) -> dict:
    sl = df.loc[start - pd.Timedelta(days=warmup): end]
    sl = sl[sl.index < end]
    logp = np.log1p(sl["ret"]).cumsum()
    pos = STRATEGIES[name](logp)
    pos = pos.shift(1).fillna(0.0)               # исполнение на следующем баре
    in_win = sl.index >= start
    pos = pos.where(in_win, 0.0)
    gross = pos * sl["ret"]
    turnover = pos.diff().abs().fillna(0.0)
    roll_cost = (sl["roll"] & (pos != 0)).astype(float) * 2 * abs(pos)   # закрыть+открыть
    cost = (turnover + roll_cost) * fee
    net = (gross - cost)[in_win]
    trades = int(((turnover > 0) & in_win).sum())
    eq = net.cumsum()
    dd = (eq - eq.cummax()).min() if len(eq) else 0.0
    return {"gross": gross[in_win].sum(), "net": net.sum(), "fees": cost[in_win].sum(),
            "trades": trades, "maxdd": dd}


# ---------- трендовость окна

def trend_metrics(sl: pd.DataFrame) -> dict:
    logp = np.log1p(sl["ret"]).cumsum()
    h4 = logp.resample("4h").last().dropna()
    path = h4.diff().abs().sum()
    er = abs(h4.iloc[-1] - h4.iloc[0]) / path if path > 0 else np.nan
    x = np.arange(len(logp))
    r = np.corrcoef(x, logp.values)[0, 1]
    daily = logp.resample("1D").last().dropna()
    dret = daily.diff().dropna()
    return {"ER": er, "R2": r * r, "move": float(logp.iloc[-1] - logp.iloc[0]),
            "dir": "up" if logp.iloc[-1] > logp.iloc[0] else "down",
            "vol_d": float(dret.std()) if len(dret) > 2 else np.nan, "bars": len(sl)}


def scan(df: pd.DataFrame, window: int = WINDOW_DAYS, step: int = STEP_DAYS,
         warmup: int = WARMUP_DAYS) -> pd.DataFrame:
    t0 = df.index.min().normalize() + pd.Timedelta(days=warmup)
    t_end = df.index.max()
    rows, start = [], t0
    while start + pd.Timedelta(days=window) <= t_end + pd.Timedelta(days=1):
        end = start + pd.Timedelta(days=window)
        sl = df.loc[start:end]
        sl = sl[sl.index < end]
        if len(sl) < 200:
            start += pd.Timedelta(days=step)
            continue
        pre = df.loc[start - pd.Timedelta(days=warmup): start]
        pre = pre[pre.index < start]
        row = {"start": start, "end": end, **trend_metrics(sl)}
        if len(pre) > 200:
            m = trend_metrics(pre)
            row.update({"pre_ER": m["ER"], "pre_R2": m["R2"]})
        res = {n: run_strategy(df, n, start, end, warmup) for n in STRATEGIES}
        row["net_median"] = float(np.median([r["net"] for r in res.values()]))
        row["net_mean"] = float(np.mean([r["net"] for r in res.values()]))
        row["net_best"] = max(r["net"] for r in res.values())
        row["share_profitable"] = float(np.mean([r["net"] > 0 for r in res.values()]))
        row["fees_mean"] = float(np.mean([r["fees"] for r in res.values()]))
        row["trades_mean"] = float(np.mean([r["trades"] for r in res.values()]))
        for n, r in res.items():
            row[f"net_{n}"] = r["net"]
        rows.append(row)
        start += pd.Timedelta(days=step)
    out = pd.DataFrame(rows)
    out["score"] = (out["ER"].rank(pct=True) + out["R2"].rank(pct=True)) / 2
    return out


def pick_non_overlapping(w: pd.DataFrame, k: int, col: str = "score", min_gap_days: int = 0) -> pd.DataFrame:
    chosen = []
    for _, r in w.sort_values(col, ascending=False).iterrows():
        if all(r["end"] <= c["start"] - pd.Timedelta(days=min_gap_days) or
               r["start"] >= c["end"] + pd.Timedelta(days=min_gap_days) for c in chosen):
            chosen.append(r)
        if len(chosen) == k:
            break
    return pd.DataFrame(chosen).sort_values("start")
