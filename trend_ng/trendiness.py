"""Проверка гипотезы: есть ли у NGM явная трендовость (погода, сезонность, спекулянты)?

1. Variance ratio VR(q) на 15-минутных барах: >1 тренд, <1 возврат к среднему. Нуль - перемешанные доходности.
2. Автокорреляция дневных доходностей (лаги 1..5), пул по контрактам и по группам.
3. Efficiency ratio торгового месяца: реальные контракты против суррогатов (перемешанные дни).
4. Доходность торгового месяца по календарным месяцам экспирации (сезонность).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from study import Contract, load_contracts, make_surrogate, resample, split_groups

REPORTS = Path(__file__).resolve().parent.parent / "reports"
pd.set_option("display.width", 250)
rng = np.random.default_rng(7)
cts = load_contracts()
groups = split_groups(cts)
gid = {s: k + 1 for k, g in enumerate(groups) for s in g}


def trade_returns_15m(ct: Contract) -> np.ndarray:
    o, c, it, _ = resample(ct, 15)
    r = np.diff(np.log(c))[it[1:]]
    return r


def vr(r: np.ndarray, q: int) -> float:
    cs = np.r_[0.0, np.cumsum(r)]
    rq = cs[q:] - cs[:-q]
    return rq.var(ddof=1) / (q * r.var(ddof=1))


HORIZONS = {"1ч": 4, "4ч": 16, "1д(~58)": 58, "2д": 116, "5д": 290}
print("== 1. Variance ratio, 15-минутные бары, торговый месяц ==")
rows = []
for ct in cts:
    r = trade_returns_15m(ct)
    rows.append({"secid": ct.secid, "group": gid[ct.secid], **{h: vr(r, q) for h, q in HORIZONS.items()}})
vrdf = pd.DataFrame(rows)
null = np.array([[np.mean([vr(rng.permutation(trade_returns_15m(ct)), q) for ct in cts]) for q in HORIZONS.values()]
                 for _ in range(300)])
tab = pd.DataFrame({"VR реальный (среднее по 22)": vrdf[list(HORIZONS)].mean(),
                    "нуль: среднее": null.mean(axis=0), "нуль: 2.5%": np.quantile(null, 0.025, axis=0),
                    "нуль: 97.5%": np.quantile(null, 0.975, axis=0)})
tab["доля контрактов VR>1"] = (vrdf[list(HORIZONS)] > 1).mean()
print(tab.round(3).to_string())
print("\nVR по группам:")
print(vrdf.groupby("group")[list(HORIZONS)].mean().round(3).to_string())
tab.to_csv(REPORTS / "trend_variance_ratio.csv")

print("\n== 2. Автокорреляция дневных доходностей (дни с >=300 минутных баров) ==")
def daily_returns(ct: Contract) -> np.ndarray:
    s = pd.Series(ct.c, index=ct.ts)
    d = s.groupby(s.index.normalize()).agg(["last", "size"])
    d = d[d["size"] >= 300]
    return np.diff(np.log(d["last"].to_numpy()))

def pooled_acf(rets: list[np.ndarray], lag: int) -> tuple[float, int]:
    num = den = 0.0
    n = 0
    for r in rets:
        r = r - r.mean()
        num += (r[lag:] * r[:-lag]).sum()
        den += (r * r).sum()
        n += len(r) - lag
    return num / den, n

dr = {c.secid: daily_returns(c) for c in cts}
rows = []
for lag in range(1, 6):
    a, n = pooled_acf(list(dr.values()), lag)
    nulls = [pooled_acf([rng.permutation(r) for r in dr.values()], lag)[0] for _ in range(500)]
    rows.append({"лаг": lag, "acf": a, "n": n, "нуль 2.5%": np.quantile(nulls, .025), "нуль 97.5%": np.quantile(nulls, .975),
                 "z": a / np.std(nulls)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nacf лаг 1 по группам:")
for k, g in enumerate(groups, 1):
    a, n = pooled_acf([dr[s] for s in g], 1)
    print(f"  группа {k} ({g[0]}..{g[-1]}): acf1 = {a:+.3f}, n = {n}")
pd.DataFrame(rows).to_csv(REPORTS / "trend_daily_acf.csv", index=False)

print("\n== 3. Efficiency ratio торгового месяца (1ч бары): реальные vs суррогаты (перемешанные дни) ==")
def er_month(ct: Contract) -> float:
    o, c, it, _ = resample(ct, 60)
    x = np.log(c[it])
    return abs(x[-1] - x[0]) / np.abs(np.diff(x)).sum()
real_er = np.array([er_month(c) for c in cts])
sur = np.array([[er_month(make_surrogate(c, rng)) for c in cts] for _ in range(300)])
print(f"средний ER реальный {real_er.mean():.4f}; суррогаты {sur.mean():.4f} (95%: {np.quantile(sur.mean(1), .025):.4f}..{np.quantile(sur.mean(1), .975):.4f}); "
      f"p-value (суррогат >= реальный) {(sur.mean(1) >= real_er.mean()).mean():.3f}")

print("\n== 4. Сезонность: доходность buy&hold за торговый месяц по контрактам ==")
rows = []
for ct in cts:
    o, c, it, _ = resample(ct, 60)
    rows.append({"secid": ct.secid, "group": gid[ct.secid], "месяц экспирации": ct.expiry.month,
                 "buyhold": c[it][-1] / o[it][0] - 1, "ER": er_month(ct)})
sd = pd.DataFrame(rows)
print(sd.round(3).to_string(index=False))
sd.to_csv(REPORTS / "trend_by_contract.csv", index=False)
