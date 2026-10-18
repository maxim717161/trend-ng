"""Нулевая модель для фильтра по запасам: реальные цены, но ряд запасов сдвинут на случайное число недель
(сохраняется его автокорреляция и частота знаков, пропадает связь с ценой).
Если запасы несут информацию, реальный результат фильтра должен быть заметно лучше сдвинутых."""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from study import config_grid, evaluate, group_stats, load_contracts, set_bias_shift, split_groups

REPORTS = Path(__file__).resolve().parent.parent / "reports"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CTS = load_contracts()
GROUPS = split_groups(CTS)
IDX = [np.array([i for i, c in enumerate(CTS) if c.secid in g]) for g in GROUPS]
FULL = config_grid()
GRID = [g for g in FULL if g[4] in ("none", "dev4") and g[3] == 0.0 or g[0] == "storage" and g[4] == "dev4"]
BASE = [j for j, g in enumerate(GRID) if g[4] == "none"]
DEV = [j for j, g in enumerate(GRID) if g[4] == "dev4" and g[0] != "storage"]
KEY = {(g[0], g[1], g[2]): j for j, g in enumerate(GRID) if g[4] == "none"}
PAIR = [(j, KEY[(GRID[j][0], GRID[j][1], GRID[j][2])]) for j in DEV]
STOR = [j for j, g in enumerate(GRID) if g[0] == "storage"][0]


def stats(net: np.ndarray) -> dict:
    gs = group_stats(net, IDX)
    plus = (gs > 0).sum(axis=0)
    m = net.mean(axis=0)
    return {"n4": int((plus[DEV] >= 4).sum()), "mean_net": float(m[DEV].mean()), "best_net": float(m[DEV].max()),
            "gain_net": float(np.mean([m[j] - m[k] for j, k in PAIR])),
            "share_better": float(np.mean([m[j] > m[k] for j, k in PAIR])),
            "storage_only": float(m[STOR])}


def one(k: int) -> dict:
    set_bias_shift(k)
    return stats(evaluate(CTS, GRID))


if __name__ == "__main__":
    real = stats(evaluate(CTS, GRID))
    rng = np.random.default_rng(5)
    shifts = rng.choice(np.arange(8, 560), size=N, replace=False)
    with Pool(4) as p:
        null = pd.DataFrame(p.map(one, [int(k) for k in shifts]))
    null.to_csv(REPORTS / "storage_null.csv", index=False)
    print(f"конфигураций с фильтром dev4: {len(DEV)}; сдвигов: {N}")
    pd.set_option("display.width", 200)
    print(null.describe().round(4).to_string())
    print("\nреальные:", {k: round(v, 4) for k, v in real.items()})
    for k, v in real.items():
        print(f"  p-value (сдвиги >= реальный) {k}: {(null[k] >= v).mean():.3f}")
