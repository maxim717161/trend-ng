"""Диагностика трендовости NGM только по торговым месяцам (они не пересекаются между контрактами).

1. Variance ratio VR(q) на 15-минутных барах: >1 тренд, <1 возврат. Нуль: суррогаты со случайными знаками шагов (VR = 1).
2. Автокорреляция лага 1 неперекрывающихся баров 30..2880 минут: >0 тренд. Тот же нуль, 5 групп.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nulls import make_signflip
from study import load_contracts, resample, split_groups

REPORTS = Path(__file__).resolve().parent.parent / "reports"
rng = np.random.default_rng(31)
N_NULL = 300
cts = load_contracts()
groups = split_groups(cts)
gid = {s: k for k, g in enumerate(groups) for s in g}
pd.set_option("display.width", 250)


def lr(ct, step):
    o, c, it, _ = resample(ct, step)
    return np.diff(np.log(c))[it[1:]]


def vr(r, q):
    cs = np.r_[0.0, np.cumsum(r)]
    rq = cs[q:] - cs[:-q]
    return rq.var(ddof=1) / (q * r.var(ddof=1))


HZ = {"1ч": 4, "4ч": 16, "1д": 58, "2д": 116, "5д": 290}
print("== 1. Variance ratio, 15-мин бары, торговые месяцы, 22 контракта ==")
real = {c.secid: lr(c, 15) for c in cts}
vr_real = {h: np.array([vr(real[c.secid], q) for c in cts]) for h, q in HZ.items()}
null = {h: np.zeros(N_NULL) for h in HZ}
for i in range(N_NULL):
    sur = [make_signflip(c, rng) for c in cts]
    for h, q in HZ.items():
        null[h][i] = np.mean([vr(lr(c, 15), q) for c in sur])
rows = []
for h in HZ:
    x = vr_real[h]
    gm = [x[[i for i, c in enumerate(cts) if gid[c.secid] == k]].mean() for k in range(5)]
    rows.append({"горизонт": h, "VR среднее": x.mean(), "VR медиана": np.median(x), "нуль 2.5%": np.quantile(null[h], .025),
                 "нуль 97.5%": np.quantile(null[h], .975), "p (VR>=)": (null[h] >= x.mean()).mean(),
                 "групп с VR>1": int(np.sum(np.array(gm) > 1)), "контрактов VR>1": f"{(x > 1).sum()}/{len(x)}"})
vdf = pd.DataFrame(rows)
print(vdf.round(3).to_string(index=False))
vdf.to_csv(REPORTS / "trend_vr_signflip.csv", index=False)

print("\n== 2. Автокорреляция лага 1 неперекрывающихся баров (торговые месяцы) ==")
def acf1(rets):
    num = sum(((r - r.mean())[1:] * (r - r.mean())[:-1]).sum() for r in rets)
    den = sum(((r - r.mean()) ** 2).sum() for r in rets)
    return num / den
rows = []
for st in (30, 60, 120, 240, 480, 1440, 2880):
    rr = {c.secid: lr(c, st) for c in cts}
    rr = {k: v for k, v in rr.items() if len(v) > 4}
    a = acf1(list(rr.values()))
    nl = [acf1([lr(make_signflip(c, rng), st) for c in cts if c.secid in rr]) for _ in range(200)]
    gs = [acf1([rr[s] for s in g if s in rr]) for g in groups]
    rows.append({"бар, мин": st, "баров": sum(len(v) for v in rr.values()), "acf1": a, "z": a / np.std(nl),
                 "p (>=)": (np.array(nl) >= a).mean(), "групп с acf1>0": int(np.sum(np.array(gs) > 0)),
                 **{f"G{k+1}": v for k, v in enumerate(gs)}})
adf = pd.DataFrame(rows)
print(adf.round(3).to_string(index=False))
adf.to_csv(REPORTS / "trend_acf_signflip.csv", index=False)
