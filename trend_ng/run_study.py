from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study import FAMILIES, MIN_WARM_BARS, WARM_MONTHS, backtest, load_contracts, summarize, sweep

MIN_TRAIN = 5
REPORTS = Path(__file__).resolve().parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

cts = load_contracts(verbose=True)
print(f"прогрев {WARM_MONTHS} мес., минимум {MIN_WARM_BARS} баров; контрактов: {len(cts)}: {[c.secid for c in cts]}")
res = sweep(cts)
res.to_csv(REPORTS / "sweep_by_contract.csv", index=False)
summ = summarize(res)
summ.to_csv(REPORTS / "sweep_summary.csv", index=False)

bh = res.drop_duplicates("secid")[["secid", "buyhold"]]
print(f"buy&hold за торговый месяц: среднее {bh.buyhold.mean():+.3f}, медиана {bh.buyhold.median():+.3f}, |среднее| {bh.buyhold.abs().mean():.3f}")

print("\nБольшие окна (n >= 1440):")
print(summ[summ.n >= 1440][["family", "n", "net_mean", "net_median", "t_stat", "trades_mean", "coverage", "win_rate"]]
      .round(4).to_string(index=False))

print("\nЛучшие окна по каждому семейству (по среднему net на контракт):")
best = summ.loc[summ.groupby("family")["net_mean"].idxmax()]
print(best.round(4).to_string(index=False))
print("\nТоп-10 конфигураций по net_mean:")
print(summ.nlargest(10, "net_mean").round(4).to_string(index=False))

# 1) устойчивость рейтинга окон: первая половина контрактов vs вторая
secids = [c.secid for c in cts]
half = len(secids) // 2
first, second = secids[:half], secids[half:]
print(f"\nУстойчивость: контракты {first[0]}..{first[-1]} (A) vs {second[0]}..{second[-1]} (B)")
for f in FAMILIES:
    a = res[(res.family == f) & res.secid.isin(first)].groupby("n")["net"].mean()
    b = res[(res.family == f) & res.secid.isin(second)].groupby("n")["net"].mean()
    n_best_a = a.idxmax()
    print(f"  {f:10s} Spearman(A,B) по окнам = {a.corr(b, method='spearman'):+.2f}; "
          f"лучшее окно на A = {n_best_a} (A {a.max():+.3f}, B {b[n_best_a]:+.3f}); "
          f"лучшее на B = {b.idxmax()} (B {b.max():+.3f}, A {a[b.idxmax()]:+.3f})")

# 2) walk-forward: на каждом контракте выбираем окно по среднему net предыдущих контрактов
print(f"\nWalk-forward (расширяющаяся выборка, минимум {MIN_TRAIN} контрактов обучения):")
piv = res.pivot_table(index="secid", columns=["family", "n"], values="net")
piv = piv.loc[[c.secid for c in cts]]
wf_rows = []
for fam in list(FAMILIES) + ["all"]:
    cols = [c for c in piv.columns if fam == "all" or c[0] == fam]
    out = []
    for k in range(MIN_TRAIN, len(piv)):
        train = piv.iloc[:k][cols].mean()
        pick = train.idxmax()
        out.append((piv.index[k], pick, piv.iloc[k][pick]))
    o = pd.DataFrame(out, columns=["secid", "pick", "net"])
    wf_rows.append((fam, o))
    print(f"  {fam:10s} тестовых контрактов {len(o)}: среднее net {o.net.mean():+.4f}, "
          f"медиана {o.net.median():+.4f}, доля плюсовых {(o.net > 0).mean():.0%}; "
          f"выбранные окна: {[int(p[1]) for p in o['pick']]}")
wf_rows[-1][1].to_csv(REPORTS / "walk_forward_all.csv", index=False)

# 3) лучшая по подвыборке конфигурация по контрактам
fam, n = best.loc[best["net_mean"].idxmax(), ["family", "n"]]
print(f"\nПо контрактам: {fam} n={int(n)}")
d = res[(res.family == fam) & (res.n == n)][["secid", "expiry", "gross", "fees", "net", "trades", "buyhold"]]
print(d.assign(expiry=d.expiry.dt.date).round(4).to_string(index=False))

# 4) чувствительность к комиссии для лучших окон
print("\nЧувствительность к комиссии (среднее net на контракт):")
for f, nn in best[["family", "n"]].itertuples(index=False):
    row = [f"{f} n={nn}"]
    for fee in (0.0, 0.0002, 0.0005, 0.0010):
        row.append(f"fee={fee:.2%}: {np.mean([backtest(c, f, int(nn), fee)['net'] for c in cts]):+.4f}")
    print("  " + "  ".join(row))

# 5) зависимость от одного контракта и плато
print("\nБез NRF6 (январь 2026, +79% buy&hold):")
for f, nn in best[["family", "n"]].itertuples(index=False):
    x = res[(res.family == f) & (res.n == nn)]
    print(f"  {f} n={nn}: с NRF6 {x.net.mean():+.4f}, без NRF6 {x[x.secid != 'NRF6'].net.mean():+.4f}")
print("\nПлато: скользящее среднее net по 3 соседним значениям сетки")
for f in ("donchian", "ema_cross"):
    sm = summ[summ.family == f].set_index("n")["net_mean"]
    r = sm.rolling(3, center=True).mean().dropna()
    print(f"  {f}: лучшее сглаженное окно n={r.idxmax()} ({r.max():+.4f}); сглаженные: " + ", ".join(f"{k}:{v:+.3f}" for k, v in r.items() if k >= 60))

# графики
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=False)
for f in FAMILIES:
    s = summ[summ.family == f]
    axes[0].plot(s.n, s.net_mean, marker="o", label=f)
    axes[1].plot(s.n, s.gross_mean, marker="o", label=f)
for ax, t in zip(axes, ["net (после комиссии 0.05%/сторона)", "gross (до комиссии)"]):
    ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.6); ax.set_title(f"средний результат за торговый месяц, {t}")
    ax.set_xlabel("размер окна, минут"); ax.legend(); ax.set_ylim(-0.5, 0.15); ax.set_xlim(4, 20000)
fig.tight_layout(); fig.savefig(REPORTS / "window_sizes.png", dpi=110)
