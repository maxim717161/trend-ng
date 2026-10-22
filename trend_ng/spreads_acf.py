"""Автокорреляция доходностей спреда (lag 1, непересекающиеся бары) на торговых месяцах: сам спред vs ближний контракт.
Отрицательная автокорреляция на барах 15 минут и меньше у спреда двух редко торгуемых ног - признак микроструктуры
(отскок цены между bid и ask), а не экономического возврата к среднему."""
import numpy as np
import pandas as pd

from spreads import INFO, SPS
from study import load_contracts, resample, split_groups

NEAR = {c.secid: c for c in load_contracts()}
STEPS = [5, 15, 60, 240, 720, 1440]


def acf1(o, c, it):
    r = (c / np.r_[o[0], c[:-1]] - 1)[it]
    r = r - r.mean()
    return float((r[1:] * r[:-1]).sum() / (r ** 2).sum()), len(r)


rows = []
for h in STEPS:
    for name, items in (("спред", {s.secid: s for s in SPS}), ("ближний", NEAR)):
        v = []
        for ct in items.values():
            o, c, it, _ = resample(ct, h)
            a, n = acf1(o, c, it)
            v.append((a, n))
        a = np.array([x[0] for x in v]); n = np.array([x[1] for x in v])
        pooled = (a * n).sum() / n.sum()
        rows.append({"бар, мин": h, "ряд": name, "средняя acf1": pooled, "t (приближённо)": pooled * np.sqrt(n.sum()),
                     "контрактов с acf<0": int((a < 0).sum()), "баров всего": int(n.sum())})
df = pd.DataFrame(rows)
df.to_csv("../reports/spreads_acf.csv", index=False)
print(df.round(3).to_string(index=False))
