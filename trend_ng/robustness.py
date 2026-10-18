"""Проверка найденных (>=4 из 5 групп) конфигураций на устойчивость + графики."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study import config_grid, evaluate, load_contracts

REPORTS = Path(__file__).resolve().parent.parent / "reports"
cts = load_contracts()
grid = config_grid()
net = evaluate(cts, grid)
secids = [c.secid for c in cts]
res = pd.read_csv(REPORTS / "search_results.csv")
passed = np.flatnonzero(res.plus_groups.to_numpy() >= 4)
print(f"прошли критерий >=4/5: {len(passed)} конфигураций")


def plus_groups(mat: np.ndarray, shift: int = 0, drop: tuple[str, ...] = ()) -> np.ndarray:
    keep = [i for i, s in enumerate(secids) if s not in drop]
    order = keep[shift:] + keep[:shift] if shift else keep   # циклический сдвиг границ групп
    parts = np.array_split(np.array(order), 5)
    return np.vstack([mat[p].mean(axis=0) for p in parts])


print("\nПроверка прошедших конфигураций при других границах групп и без NRF6:")
rows = []
for j in passed:
    r = {"config": f"{grid[j][0]} n={grid[j][1]} step={grid[j][2]} er={grid[j][3]}"}
    for sh in (0, 2, 4, 6):
        # сдвиг без циклического переноса хронологии: убираем первые sh контрактов
        m = net[sh:, j:j + 1]
        parts = np.array_split(np.arange(m.shape[0]), 5)
        r[f"drop_first_{sh}"] = int(sum(m[p].mean() > 0 for p in parts))
    keep = [i for i, s in enumerate(secids) if s != "NRF6"]
    parts = np.array_split(np.array(keep), 5)
    r["without_NRF6"] = int(sum(net[p][:, j].mean() > 0 for p in parts))
    rows.append(r)
rob = pd.DataFrame(rows)
print(rob.to_string(index=False))
print(f"\nустойчиво (>=4 из 5 во всех проверках): {int((rob.drop(columns='config') >= 4).all(axis=1).sum())} из {len(rob)}")
rob.to_csv(REPORTS / "robustness_passed.csv", index=False)

sur = pd.read_csv(REPORTS / "surrogates.csv")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].hist(sur.n4, bins=30, color="lightgray", edgecolor="k")
ax[0].axvline((res.plus_groups >= 4).sum(), color="r", lw=2, label=f"реальные данные: {(res.plus_groups >= 4).sum()}")
ax[0].set_title("Сколько из 606 конфигураций дают плюс в >=4 из 5 групп\n(суррогаты: перемешанные дни, 200 шт.)")
ax[0].legend()
vr = pd.read_csv(REPORTS / "trend_variance_ratio.csv", index_col=0)
x = np.arange(len(vr))
ax[1].fill_between(x, vr["нуль: 2.5%"], vr["нуль: 97.5%"], color="lightgray", label="нуль (95%)")
ax[1].plot(x, vr.iloc[:, 0], "ro-", label="реальный VR"); ax[1].axhline(1, color="k", lw=0.6)
ax[1].set_xticks(x, vr.index); ax[1].set_title("Variance ratio (15-мин бары): >1 тренд, <1 возврат"); ax[1].legend()
fig.tight_layout(); fig.savefig(REPORTS / "trendiness.png", dpi=110)
