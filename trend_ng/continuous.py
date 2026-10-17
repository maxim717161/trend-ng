"""Склейка непрерывного ряда NGM из месячных контрактов.

Фронт-контракт на дату d выбирается по объёму за предыдущий торговый день
(без заглядывания вперёд), экспирация фронта только растёт.
Доходности считаются внутри одного контракта; в момент ролла новый
контракт берёт доходность от своей же последней цены, поэтому разрывов
между контрактами в ряде нет (аналог ratio-adjust).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    bars, daily = [], []
    for p in sorted(RAW.glob("*_60m.parquet")):
        bars.append(pd.read_parquet(p))
    for p in sorted(RAW.glob("*_daily.parquet")):
        daily.append(pd.read_parquet(p))
    bars = pd.concat(bars, ignore_index=True)
    daily = pd.concat(daily, ignore_index=True)
    daily["TRADEDATE"] = pd.to_datetime(daily["TRADEDATE"])
    return bars, daily


def contract_expiry(bars: pd.DataFrame) -> pd.Series:
    return bars.groupby("secid")["begin"].max()


def pick_front(daily: pd.DataFrame, expiry: pd.Series) -> pd.Series:
    """Для каждой календарной даты d -> secid фронта по данным до d."""
    vol = daily.pivot_table(index="TRADEDATE", columns="SECID", values="VOLUME", aggfunc="sum")
    dates = vol.index
    chosen, cur_exp = {}, pd.Timestamp.min
    prev_choice = None
    for i, d in enumerate(dates):
        row = vol.iloc[i].dropna()
        row = row[[s for s in row.index if expiry.get(s, pd.Timestamp.min) >= d]]
        if row.empty:
            continue
        best = row.idxmax()
        if expiry[best] < cur_exp:
            cands = row[[s for s in row.index if expiry[s] >= cur_exp]]
            best = cands.idxmax() if not cands.empty else prev_choice
        cur_exp = max(cur_exp, expiry[best])
        prev_choice = best
        if i + 1 < len(dates):
            chosen[dates[i + 1]] = best
    return pd.Series(chosen, name="front")


def build() -> pd.DataFrame:
    bars, daily = load_raw()
    expiry = contract_expiry(bars)
    front = pick_front(daily, expiry)
    bars = bars.sort_values(["secid", "begin"]).reset_index(drop=True)
    bars["date"] = bars["begin"].dt.normalize()
    cal = pd.Series(front.reindex(pd.date_range(front.index.min(), bars["date"].max())).ffill())
    bars["front"] = bars["date"].map(cal)
    sel = bars[bars["secid"] == bars["front"]].sort_values("begin").copy()
    sel = sel[sel["begin"] >= front.index.min()]

    bars_idx = bars.set_index(["secid", "begin"])["close"]
    prev_close = bars.groupby("secid")["close"].shift(1)
    prev_ts = bars.groupby("secid")["begin"].shift(1)
    bars["prev_close"] = prev_close
    bars["prev_begin"] = prev_ts
    sel = sel.merge(bars[["secid", "begin", "prev_close", "prev_begin"]], on=["secid", "begin"], how="left")
    sel["ret"] = sel["close"] / sel["prev_close"] - 1
    # разрыв > 5 дней внутри контракта (нет торгов) - не считаем доходность
    gap = (sel["begin"] - sel["prev_begin"]).dt.days > 5
    sel.loc[gap, "ret"] = np.nan
    sel["roll"] = (sel["secid"] != sel["secid"].shift(1)) & sel["secid"].shift(1).notna()
    sel["ret"] = sel["ret"].fillna(0.0)
    sel["logp"] = np.log1p(sel["ret"]).cumsum()
    sel["price"] = np.exp(sel["logp"])
    # прогнозные high/low/close в той же адаптированной шкале
    scale = sel["price"] / sel["close"]
    sel["adj_close"] = sel["close"] * scale
    sel["adj_high"] = sel["high"] * scale
    sel["adj_low"] = sel["low"] * scale
    sel["adj_open"] = sel["open"] * scale
    cols = ["begin", "secid", "open", "high", "low", "close", "volume", "ret", "roll",
            "adj_open", "adj_high", "adj_low", "adj_close"]
    out = sel[cols].reset_index(drop=True)
    OUT.mkdir(exist_ok=True)
    out.to_parquet(OUT / "ngm_continuous_60m.parquet")
    return out


if __name__ == "__main__":
    df = build()
    print(df.shape, df.begin.min(), df.begin.max())
    print("rolls:", int(df.roll.sum()))
    print(df[df.roll][["begin", "secid"]].to_string())
    print(df.ret.describe())
    print(df.loc[df.ret.abs().nlargest(8).index, ["begin", "secid", "ret"]])
