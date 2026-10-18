"""Прямая проверка трендовости: валовая (до комиссии) доходность трендовых стратегий по размеру окна.

Для каждого окна n усредняются трендовые стратегии (donchian, ema_cross, momentum; шаги 1-240 мин, без фильтров).
Если в NGM есть тренд, валовая доходность должна быть устойчиво > 0 и заметно выше, чем у случайного блуждания.
Показываются: среднее по 22 контрактам с bootstrap-интервалом по контрактам, без NRF6 (самый трендовый месяц),
число групп из 5 с плюсом, p-value по суррогатам со случайными знаками шагов.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from nulls import make_signflip
from study import config_grid, evaluate, load_contracts, split_groups

REPORTS = Path(__file__).resolve().parent.parent / "reports"
N_SURR = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CTS = load_contracts()
SEC = [c.secid for c in CTS]
GRID = [g for g in config_grid(storage=False) if g[3] == 0.0 and g[4] == "none"]
NS = sorted({g[1] for g in GRID})
COLS = {n: [j for j, g in enumerate(GRID) if g[1] == n] for n in NS}
GROUPS = split_groups(CTS)
IDX = [np.array([i for i, c in enumerate(CTS) if c.secid in g]) for g in GROUPS]


def gross_matrix(cts) -> np.ndarray:
    d = evaluate(cts, GRID, detail=True)
    pv = d.pivot_table(index="secid", columns=["family", "n", "step"], values="gross")
    pv = pv.loc[[c.secid for c in cts]]
    cols = [(g[0], g[1], g[2]) for g in GRID]
    return pv[cols].to_numpy()


def by_window(gm: np.ndarray) -> np.ndarray:
    """[контракт, окно]: среднее валовой доходности по семействам и шагам."""
    return np.column_stack([gm[:, COLS[n]].mean(axis=1) for n in NS])


def null_run(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return by_window(gross_matrix([make_signflip(c, rng) for c in CTS])).mean(axis=0)


if __name__ == "__main__":
    gw = by_window(gross_matrix(CTS))
    rng = np.random.default_rng(0)
    with Pool(4) as p:
        null = np.array(p.map(null_run, range(300, 300 + N_SURR)))
    pd.DataFrame(null, columns=NS).to_csv(REPORTS / "trend_gross_null.csv", index=False)
    k = SEC.index("NRF6")
    rows = []
    for j, n in enumerate(NS):
        x = gw[:, j]
        boots = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)])
        rows.append({"окно, мин": n, "среднее gross, %": x.mean() * 100,
                     "95% bootstrap": f"{np.quantile(boots, .025)*100:+.1f}..{np.quantile(boots, .975)*100:+.1f}",
                     "медиана, %": np.median(x) * 100, "без NRF6, %": np.delete(x, k).mean() * 100,
                     "групп с плюсом": int(sum(x[ix].mean() > 0 for ix in IDX)),
                     "плюсовых контрактов": f"{(x > 0).sum()}/{len(x)}",
                     "нуль: среднее, %": null[:, j].mean() * 100, "нуль: 95-й перц., %": np.quantile(null[:, j], .95) * 100,
                     "p-value": (null[:, j] >= x.mean()).mean()})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "trend_gross_by_window.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"Валовая доходность трендовых стратегий по окну (22 контракта, суррогатов {N_SURR}):")
    print(df.round(2).to_string(index=False))
    overall = gw.mean(axis=1)
    print(f"\nСреднее по всем окнам: {overall.mean()*100:+.2f}% на контракт (без NRF6 {np.delete(overall, k).mean()*100:+.2f}%), "
          f"нуль {null.mean()*100:+.2f}% (95-й перцентиль среднего по окнам {np.quantile(null.mean(axis=1), .95)*100:+.2f}%), "
          f"p = {(null.mean(axis=1) >= gw.mean()).mean():.3f}")
    print(f"Максимум по окнам: реальный {gw.mean(axis=0).max()*100:+.2f}% (окно {NS[int(gw.mean(axis=0).argmax())]}), "
          f"нуль: среднее максимума {null.max(axis=1).mean()*100:+.2f}%, 95-й перцентиль {np.quantile(null.max(axis=1), .95)*100:+.2f}%, "
          f"p (с поправкой на выбор окна) = {(null.max(axis=1) >= gw.mean(axis=0).max()).mean():.3f}")
    mid = [j for j, n in enumerate(NS) if 240 <= n <= 5760]
    print(f"Окна 240-5760: {gw[:, mid].mean()*100:+.2f}%, нуль {null[:, mid].mean()*100:+.2f}%, "
          f"p = {(null[:, mid].mean(axis=1) >= gw[:, mid].mean()).mean():.3f}")
