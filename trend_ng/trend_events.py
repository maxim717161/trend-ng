"""Продолжение движения после крупных ходов (событийная трендовая стратегия), только торговые месяцы контрактов.

Триггер: за последние w минут цена сдвинулась на >= k сигм (сигма - std таких сдвигов по прошлым 4000 пятиминуткам).
Вход по open следующей 5-минутки в сторону движения, удержание h минут, комиссия 0.05% за сторону.
События не перекрываются. Группы - 5 хронологических групп контрактов, критерий - плюс в >= 4 группах.
Нуль: суррогаты со случайными знаками шагов (события ищутся заново на каждом суррогате).
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from nulls import make_signflip
from study import FEE, load_contracts, resample, split_groups

REPORTS = Path(__file__).resolve().parent.parent / "reports"
N_SURR = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CTS = load_contracts()
GROUPS = split_groups(CTS)
GID = {s: k for k, g in enumerate(GROUPS) for s in g}
WS, KS, HS = (60, 240, 1440), (1.5, 2.5, 3.5), (15, 60, 240, 1440)
CELLS = [(w, k, h) for w in WS for k in KS for h in HS]


def contract_events(ct) -> dict:
    o, c, it, _ = resample(ct, 5)
    lc = np.log(c)
    out = {cell: [] for cell in CELLS}
    for w in WS:
        bw = w // 5
        mv = np.r_[np.full(bw, np.nan), lc[bw:] - lc[:-bw]]
        sig = pd.Series(mv).rolling(4000, min_periods=1000).std().shift(1).to_numpy()
        for k in KS:
            trig = np.flatnonzero(it & np.isfinite(sig) & (np.abs(mv) >= k * sig))
            for h in HS:
                hb = h // 5
                nxt = -1
                for t in trig:
                    if t <= nxt or t + 1 >= len(c):
                        continue
                    end = min(t + hb, len(c) - 1)
                    d = np.sign(mv[t])
                    gross = d * (c[end] / o[t + 1] - 1)
                    out[(w, k, h)].append((gross, gross - 2 * FEE))
                    nxt = t + max(bw, hb)
    return out


def stats(cts) -> pd.DataFrame:
    ev = {cell: [] for cell in CELLS}
    for ct in cts:
        res = contract_events(ct)
        for cell in CELLS:
            ev[cell] += [(GID[ct.secid], g, n) for g, n in res[cell]]
    rows = []
    for cell in CELLS:
        a = np.array(ev[cell]).reshape(-1, 3)
        if len(a) == 0:
            rows.append({"w": cell[0], "k": cell[1], "h": cell[2], "n": 0, "gross_bp": np.nan, "net_bp": np.nan, "t": np.nan, "plus_groups": 0})
            continue
        gm = [a[a[:, 0] == q, 2].mean() if (a[:, 0] == q).any() else np.nan for q in range(5)]
        rows.append({"w": cell[0], "k": cell[1], "h": cell[2], "n": len(a), "gross_bp": a[:, 1].mean() * 1e4,
                     "net_bp": a[:, 2].mean() * 1e4, "t": a[:, 2].mean() / (a[:, 2].std(ddof=1) / np.sqrt(len(a))) if len(a) > 2 else np.nan,
                     "plus_groups": int(np.sum(np.array(gm) > 0)), **{f"G{q+1}_bp": gm[q] * 1e4 for q in range(5)}})
    return pd.DataFrame(rows)


def null_run(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    d = stats([make_signflip(c, rng) for c in CTS])
    return {"n4": int((d.plus_groups >= 4).sum()), "max_net": float(d.net_bp.max()), "max_t": float(d.t.max()),
            "mean_gross": float(d.gross_bp.mean())}


if __name__ == "__main__":
    real = stats(CTS)
    real.to_csv(REPORTS / "trend_events.csv", index=False)
    pd.set_option("display.width", 250)
    print("Продолжение движения после крупных ходов, торговые месяцы 22 контрактов (комиссия круга 10 б.п.):")
    print(real.round(2).to_string(index=False))
    with Pool(4) as p:
        null = pd.DataFrame(p.map(null_run, range(500, 500 + N_SURR)))
    null.to_csv(REPORTS / "trend_events_null.csv", index=False)
    r = {"n4": int((real.plus_groups >= 4).sum()), "max_net": float(real.net_bp.max()), "max_t": float(real.t.max()),
         "mean_gross": float(real.gross_bp.mean())}
    print(f"\nСуррогаты ({N_SURR}):")
    print(null.describe().round(2).to_string())
    print("реальные:", {k: round(v, 2) for k, v in r.items()})
    for k, v in r.items():
        print(f"  p-value {k}: {(null[k] >= v).mean():.3f}")
