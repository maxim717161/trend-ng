"""Проверка моментума (time-series momentum): знак доходности за последние L минут задаёт позицию на следующие H минут.

Сетка:
  H (шаг пересмотра позиции) - 15, 60, 240, 720, 1440 минут (бары по H минут; 1440 = суточные бары);
  L (глубина оглядывания) - 60..10000 минут, L >= H;
  thr - порог значимости: 0 = всегда по знаку; 1 = менять позицию, только если |доходность L| > 1 сигмы
        (сигма из доходностей баров за последние ~10000 минут), иначе держим прежнюю позицию.
Всего 96 ячеек. Каждая считается по всем 22 контрактам, торговля в последний месяц, комиссия 0.05% за сторону.

Нуль: суррогаты со случайными знаками шагов (nulls.make_signflip) - у них любая зависимость от прошлого равна нулю.
Все выводы с поправкой на множественные проверки: сравнение максимума/числа проходящих ячеек с суррогатами.

Дополнительно:
  - месячный моментум: знак доходности за L месяцев до торгового месяца -> позиция на весь торговый месяц;
  - сезонность: корзина дневных моментум-ячеек по сезону экспирации (зима/лето/межсезонье).
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from nulls import make_signflip
from study import FEE, Contract, load_contracts, resample, split_groups

REPORTS = Path(__file__).resolve().parent.parent / "reports"
N_SURR = int(sys.argv[1]) if len(sys.argv) > 1 else 200
STEPS = [15, 60, 240, 720, 1440]
LOOKBACKS = [60, 120, 240, 480, 720, 960, 1440, 2880, 4320, 5760, 7200, 10000]
THRESHOLDS = [0.0, 1.0]
SIGMA_SPAN = 10_000
MONTHS = [1, 2, 3]
GRID = [(h, L, t) for h in STEPS for L in LOOKBACKS if L >= h for t in THRESHOLDS]

CTS = load_contracts()
SEC = [c.secid for c in CTS]
GROUPS = split_groups(CTS)
IDX = [np.array([i for i, c in enumerate(CTS) if c.secid in g]) for g in GROUPS]
SEASON = np.array([{12: "зима", 1: "зима", 2: "зима", 6: "лето", 7: "лето", 8: "лето"}.get(c.expiry.month, "межсезонье")
                   for c in CTS])
BASKET = np.array([h >= 60 and L >= 240 and t == 0.0 for h, L, t in GRID])


def momentum_signal(c: np.ndarray, k: int, thr: float, h: int) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(c)
    ret = (s / s.shift(k) - 1).to_numpy()
    if thr == 0.0:
        return np.nan_to_num(np.sign(ret)), np.arange(len(c)) + 1 >= k + 1
    bar = s.pct_change()
    sd = bar.rolling(max(20, SIGMA_SPAN // h), min_periods=20).std().to_numpy() * np.sqrt(k)
    z = ret / sd
    sig = pd.Series(np.where(z > thr, 1.0, np.where(z < -thr, -1.0, np.nan))).ffill().fillna(0.0).to_numpy()
    return sig, (np.arange(len(c)) + 1 >= k + 1) & np.isfinite(sd)


def pnl(o, c, in_trade, sig, ready) -> tuple[float, float, int]:
    sig = np.where(in_trade & ready, sig, 0.0)
    pos_new = np.r_[0.0, sig[:-1]]
    pos_new[-1] = 0.0
    pos_old = np.r_[0.0, pos_new[:-1]]
    gross = pos_old * (o / np.r_[c[0], c[:-1]] - 1) + pos_new * (c / o - 1)
    turn = np.abs(pos_new - pos_old)
    g = gross[in_trade].sum()
    return g, g - FEE * turn[in_trade].sum(), int(((turn > 0) & in_trade).sum())


def cell_results(ct: Contract) -> np.ndarray:
    """[ячейка, (gross, net, trades)]"""
    out = np.zeros((len(GRID), 3))
    cache = {h: resample(ct, h) for h in STEPS}
    for j, (h, L, t) in enumerate(GRID):
        o, c, it, _ = cache[h]
        sig, ready = momentum_signal(c, L // h, t, h)
        out[j] = pnl(o, c, it, sig, ready)
    return out


def monthly_results(ct: Contract) -> np.ndarray:
    """[L месяцев, (прошлая доходность, доходность торгового месяца)], NaN если истории не хватает."""
    out = np.full((len(MONTHS), 2), np.nan)
    i0 = int(ct.ts.searchsorted(ct.trade_start))
    if i0 < 1 or i0 >= len(ct.c) - 1:
        return out
    fwd = ct.c[-1] / ct.o[i0] - 1
    for j, m in enumerate(MONTHS):
        t_ref = ct.trade_start - pd.DateOffset(months=m)
        i_ref = int(ct.ts.searchsorted(t_ref))
        if ct.ts[0] > t_ref + pd.Timedelta(days=5) or i_ref >= i0:
            continue
        out[j] = (ct.c[i0 - 1] / ct.c[i_ref] - 1, fwd)
    return out


def run_all(cts: list[Contract]) -> tuple[np.ndarray, np.ndarray]:
    return (np.stack([cell_results(c) for c in cts]), np.stack([monthly_results(c) for c in cts]))


def null_run(seed: int):
    rng = np.random.default_rng(seed)
    return run_all([make_signflip(c, rng) for c in CTS])


def monthly_stats(mr: np.ndarray) -> np.ndarray:
    """[L месяцев, (средний gross, доля угаданных знаков, n)] для позиции sign(past) на весь торговый месяц."""
    res = np.full((len(MONTHS), 3), np.nan)
    for j in range(len(MONTHS)):
        ok = np.isfinite(mr[:, j, 0])
        g = np.sign(mr[ok, j, 0]) * mr[ok, j, 1]
        res[j] = (g.mean(), (g > 0).mean(), ok.sum())
    return res


def season_stats(cells: np.ndarray) -> np.ndarray:
    b = cells[:, BASKET, 0].mean(axis=1)
    return np.array([b[SEASON == s].mean() for s in ("зима", "лето", "межсезонье")] + [b.mean()])


if __name__ == "__main__":
    real_cells, real_month = run_all(CTS)
    with Pool(4) as p:
        res = p.map(null_run, range(1000, 1000 + N_SURR))
    nc = np.stack([r[0] for r in res])  # [суррогат, контракт, ячейка, метрика]
    nm = np.stack([r[1] for r in res])
    np.save(REPORTS / "momentum_null_cells.npy", nc[:, :, :, :2].mean(axis=1))

    gross, net, trades = (real_cells[:, :, i] for i in range(3))
    ng, nn = nc[:, :, :, 0], nc[:, :, :, 1]
    gm, nm_ = gross.mean(axis=0), net.mean(axis=0)
    ngm, nnm = ng.mean(axis=1), nn.mean(axis=1)

    def groups_plus(x):  # x [контракт, ячейка] -> число групп со средним > 0
        return sum((x[ix].mean(axis=0) > 0).astype(int) for ix in IDX)

    gp = groups_plus(net)
    ngp = np.stack([groups_plus(nn[s]) for s in range(N_SURR)])
    k = SEC.index("NRF6")
    rows = []
    for j, (h, L, t) in enumerate(GRID):
        rows.append({"шаг H": h, "окно L": L, "порог": t, "gross, %": gm[j] * 100, "gross без NRF6, %": np.delete(gross[:, j], k).mean() * 100,
                     "net, %": nm_[j] * 100, "сделок/мес": trades[:, j].mean(), "групп net>0": int(gp[j]),
                     "gross: контрактов с плюсом": int((gross[:, j] > 0).sum()),
                     "p gross": (ngm[:, j] >= gm[j]).mean(), "нуль gross, %": ngm[:, j].mean() * 100})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "momentum_cells.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)

    print(f"Моментум: {len(GRID)} ячеек, 22 контракта, суррогатов {N_SURR}\n")
    for t in THRESHOLDS:
        sub = df[df["порог"] == t]
        print(f"gross, % в месяц (порог {t}): строки L, столбцы H")
        print(sub.pivot(index="окно L", columns="шаг H", values="gross, %").round(1).to_string())
        print(f"p-value gross (порог {t}):")
        print(sub.pivot(index="окно L", columns="шаг H", values="p gross").round(3).to_string())
        print(f"net, % (порог {t}):")
        print(sub.pivot(index="окно L", columns="шаг H", values="net, %").round(1).to_string(), "\n")

    print("Сводка по сетке (реальные данные vs суррогаты)")
    summ = [
        ("ячеек с net > 0 в >= 4 группах из 5", int((gp >= 4).sum()), (ngp >= 4).sum(axis=1)),
        ("ячеек с net > 0 в 5 группах", int((gp >= 5).sum()), (ngp >= 5).sum(axis=1)),
        ("ячеек со средним net > 0", int((nm_ > 0).sum()), (nnm > 0).sum(axis=1)),
        ("лучший средний net, %", nm_.max() * 100, nnm.max(axis=1) * 100),
        ("лучший средний gross, %", gm.max() * 100, ngm.max(axis=1) * 100),
        ("средний gross по всей сетке, %", gm.mean() * 100, ngm.mean(axis=1) * 100),
        ("средний gross по корзине (H>=60, L>=240, порог 0), %", gm[BASKET].mean() * 100, ngm[:, BASKET].mean(axis=1) * 100),
    ]
    srows = []
    for name, real, nul in summ:
        p = (np.asarray(nul) >= real).mean()
        srows.append({"показатель": name, "реальные": round(float(real), 2), "суррогаты: среднее": round(float(np.mean(nul)), 2),
                      "суррогаты: 95-й перц.": round(float(np.quantile(nul, .95)), 2), "p-value": round(float(p), 3)})
        print(f"  {name}: {real:.2f}; нуль {np.mean(nul):.2f} (95-й перц. {np.quantile(nul, .95):.2f}); p = {p:.3f}")
    pd.DataFrame(srows).to_csv(REPORTS / "momentum_summary.csv", index=False)
    print(f"\nЯчейка с лучшим gross: H={GRID[int(gm.argmax())][0]}, L={GRID[int(gm.argmax())][1]}, порог={GRID[int(gm.argmax())][2]}, "
          f"gross {gm.max()*100:+.2f}%, net {nm_[int(gm.argmax())]*100:+.2f}%")
    print("Лучшие по net:")
    print(df.sort_values("net, %", ascending=False).head(8).round(3).to_string(index=False))

    print("\nКорзина дневных ячеек (H>=60, L>=240, порог 0): gross по сезону экспирации (контрактов: "
          + ", ".join(f"{s} {(SEASON == s).sum()}" for s in ("зима", "лето", "межсезонье")) + ")")
    real_s = season_stats(real_cells)
    null_s = np.stack([season_stats(nc[s]) for s in range(N_SURR)])
    srows = []
    for name, r, nul in zip(("зима", "лето", "межсезонье", "все"), real_s, null_s.T):
        p = (nul >= r).mean()
        srows.append({"сезон": name, "gross, %": r * 100, "нуль: среднее, %": nul.mean() * 100, "нуль: 95-й перц., %": np.quantile(nul, .95) * 100, "p-value": p})
    sdf = pd.DataFrame(srows)
    sdf.to_csv(REPORTS / "momentum_season.csv", index=False)
    print(sdf.round(3).to_string(index=False))
    bc = real_cells[:, BASKET, 0].mean(axis=1) * 100
    print("Корзина по контрактам, gross %: " + ", ".join(f"{c.secid} {v:+.1f}" for c, v in zip(CTS, bc)))
    print(f"макс. по трём сезонам: реальный {real_s[:3].max()*100:+.2f}%, p с поправкой на выбор сезона = "
          f"{(null_s[:, :3].max(axis=1) >= real_s[:3].max()).mean():.3f}")

    print("\nМесячный моментум: позиция = знак доходности за L месяцев до торгового месяца, держим весь торговый месяц (gross)")
    rs, ns = monthly_stats(real_month), np.stack([monthly_stats(nm[s]) for s in range(N_SURR)])
    mrows = []
    for j, m in enumerate(MONTHS):
        p = (ns[:, j, 0] >= rs[j, 0]).mean()
        mrows.append({"L, мес": m, "контрактов": int(rs[j, 2]), "gross, %": rs[j, 0] * 100, "доля угаданных": rs[j, 1],
                      "нуль: среднее, %": np.nanmean(ns[:, j, 0]) * 100, "нуль: 95-й перц., %": np.nanquantile(ns[:, j, 0], .95) * 100, "p-value": p,
                      "p-value (разворот)": (ns[:, j, 0] <= rs[j, 0]).mean()})
    mdf = pd.DataFrame(mrows)
    mdf.to_csv(REPORTS / "momentum_monthly.csv", index=False)
    print(mdf.round(3).to_string(index=False))
