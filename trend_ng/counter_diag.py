"""Диагностика контртрендовости (возврат к среднему) на NGM, контракты не склеиваются.

1. Автокорреляция неперекрывающихся доходностей на барах 1..1440 минут (пул по контрактам, 5 групп).
   Отрицательная = контртренд. Нуль: перемешивание доходностей внутри контракта.
2. «Откат последнего бара»: валовая и чистая (комиссия 0.05% за сторону) доходность стратегии -sign(r_t).
3. Условный откат после больших движений: за окно w движение >= k сигм -> доходность против движения за h.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from study import FEE, load_contracts, resample, split_groups

REPORTS = Path(__file__).resolve().parent.parent / "reports"
rng = np.random.default_rng(21)
pd.set_option("display.width", 250)
cts = load_contracts()
groups = split_groups(cts)
gid = {s: k + 1 for k, g in enumerate(groups) for s in g}
STEPS = [1, 5, 15, 30, 60, 120, 240, 480, 1440]


def returns(ct, step: int) -> np.ndarray:
    """Лог-доходности баров step минут за весь доступный период контракта, только дни с >=300 минутных баров."""
    s = pd.Series(ct.c, index=ct.ts)
    dense = s.groupby(s.index.normalize()).transform("size") >= 300
    sub = type(ct)(ct.secid, ct.expiry, ct.trade_start, ct.ts[dense.to_numpy()], ct.o[dense.to_numpy()], ct.c[dense.to_numpy()])
    if step == 1440:
        d = pd.Series(sub.c, index=sub.ts)
        last = d.groupby(d.index.normalize()).last()
        return np.diff(np.log(last.to_numpy()))
    o, c, it, first = resample(sub, step)
    return np.diff(np.log(c))


def pooled_acf(rets: list[np.ndarray], lag: int = 1) -> float:
    num = den = 0.0
    for r in rets:
        r = r[np.isfinite(r)]
        r = r - r.mean()
        num += (r[lag:] * r[:-lag]).sum()
        den += (r * r).sum()
    return num / den


print("== 1. Автокорреляция доходностей баров (лаг 1, лаг 2), все плотные дни ==")
rows = []
for st in STEPS:
    rets = {c.secid: returns(c, st) for c in cts}
    a1, a2 = pooled_acf(list(rets.values()), 1), pooled_acf(list(rets.values()), 2)
    null = [pooled_acf([rng.permutation(r) for r in rets.values()], 1) for _ in range(200)]
    by_group = [pooled_acf([rets[s] for s in g], 1) for g in groups]
    rows.append({"бар, мин": st, "n баров": sum(len(r) for r in rets.values()), "acf1": a1, "z(acf1)": a1 / np.std(null),
                 "acf2": a2, **{f"G{k+1}": v for k, v in enumerate(by_group)}, "групп с acf1<0": sum(v < 0 for v in by_group)})
acf = pd.DataFrame(rows)
print(acf.round(3).to_string(index=False))
acf.to_csv(REPORTS / "counter_acf.csv", index=False)

print("\n== 2. Откат последнего бара: позиция -sign(r_t) на следующий бар (по закрытиям, без учёта исполнения) ==")
rows = []
for st in (1, 5, 15, 30, 60, 120, 240):
    gross, cost, nb = 0.0, 0.0, 0
    for ct in cts:
        r = returns(ct, st)
        pos = -np.sign(r[:-1])
        g = pos * r[1:]
        turn = np.abs(np.diff(np.r_[0.0, pos]))
        gross += g.sum(); cost += FEE * turn.sum(); nb += len(g)
    rows.append({"бар, мин": st, "валовая за бар, б.п.": gross / nb * 1e4, "комиссия за бар, б.п.": cost / nb * 1e4,
                 "чистая за бар, б.п.": (gross - cost) / nb * 1e4})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n== 3. Откат после крупных движений: средняя доходность ПРОТИВ движения за h минут, б.п. (комиссия круг = 10 б.п.) ==")
def events(ct, w: int, k: float, hs: tuple[int, ...]):
    o, c, it, first = resample(ct, 5)
    lc = np.log(c)
    bw = w // 5
    mv = lc[bw:] - lc[:-bw]
    sig = pd.Series(mv).rolling(4000, min_periods=1000).std().shift(1).to_numpy()   # без заглядывания: сигма по прошлому
    out = []
    i = 0
    while i < len(mv):
        if np.isfinite(sig[i]) and abs(mv[i]) >= k * sig[i]:
            t = i + bw       # индекс бара, на закрытии которого движение известно
            fw = [lc[t + h // 5] - lc[t] if t + h // 5 < len(lc) else np.nan for h in hs]
            out.append((np.sign(mv[i]), *fw))
            i += bw          # без перекрытия событий
        else:
            i += 1
    return out

HS = (15, 60, 240, 1440)
rows = []
for w in (60, 240, 1440):
    for k in (1.5, 2.5, 3.5):
        ev = [(gid[c.secid], *e) for c in cts for e in events(c, w, k, HS)]
        if len(ev) < 20:
            continue
        a = np.array(ev, dtype=float)
        row = {"окно w, мин": w, "порог, сигм": k, "событий": len(a)}
        for j, h in enumerate(HS):
            x = -a[:, 1] * a[:, 2 + j]
            ok = np.isfinite(x)
            f = a[ok, 2 + j]; sg = a[ok, 1]
            null = np.array([(-sg * rng.permutation(f)).mean() for _ in range(2000)])
            row[f"h={h}: откат б.п."] = x[ok].mean() * 1e4
            row[f"h={h}: z"] = x[ok].mean() / null.std()
        rows.append(row)
cond = pd.DataFrame(rows)
print(cond.round(2).to_string(index=False))
cond.to_csv(REPORTS / "counter_conditional.csv", index=False)
