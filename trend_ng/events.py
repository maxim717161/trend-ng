"""Событийное исследование: реакция NGM на еженедельный отчёт EIA по запасам газа.

Для каждого отчёта берётся контракт NGM с максимальным объёмом за 3 часа до публикации
(контракты не склеиваются). Реакция R_h = ln(close через h минут) - ln(close перед публикацией).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import eia

DATA = Path(__file__).resolve().parent.parent / "data" / "minute"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
HORIZONS = (1, 3, 5, 10, 15, 30, 60, 120)


def load_all() -> dict[str, pd.DataFrame]:
    return {p.name.split("_")[0]: pd.read_parquet(p).set_index("begin")[["open", "close", "volume"]]
            for p in sorted(DATA.glob("*_1m.parquet"))}


def pick_contract(all_bars: dict[str, pd.DataFrame], t0: pd.Timestamp) -> str | None:
    best, best_v = None, 0.0
    for s, df in all_bars.items():
        v = df.loc[t0 - pd.Timedelta(hours=3): t0 - pd.Timedelta(minutes=1), "volume"].sum()
        if v > best_v:
            best, best_v = s, v
    return best


def build_events() -> pd.DataFrame:
    st = eia.load()
    bars = load_all()
    t_min = min(df.index.min() for df in bars.values())
    t_max = max(df.index.max() for df in bars.values())
    rows = []
    for _, r in st.iterrows():
        t0 = r["release_msk"]
        if t0 < t_min + pd.Timedelta(days=1) or t0 > t_max:
            continue
        sec = pick_contract(bars, t0)
        if sec is None:
            continue
        c = bars[sec]["close"]
        pre = c.loc[: t0 - pd.Timedelta(minutes=1)]
        if pre.empty or (t0 - pre.index[-1]) > pd.Timedelta(minutes=30):
            continue
        row = {"week_end": r["week_end"], "t0": t0, "secid": sec, "pre": pre.iloc[-1],
               "change": r["change"], "dev_change": r["dev_change"], "surprise": r["surprise"], "dev_pct": r["dev_pct"]}
        ok = True
        for h in HORIZONS:
            seg = c.loc[: t0 + pd.Timedelta(minutes=h - 1)]
            if seg.index[-1] < t0:  # нет баров после публикации
                row[f"R{h}"] = np.nan
                continue
            row[f"R{h}"] = np.log(seg.iloc[-1] / row["pre"])
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ev = build_events()
    ev.to_csv(REPORTS / "eia_events.csv", index=False)
    print(len(ev), "событий", ev.t0.min(), "..", ev.t0.max())
    print(ev[[f"R{h}" for h in HORIZONS]].describe().round(4).to_string())
