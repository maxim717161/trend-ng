"""Поиск конфигураций, дающих плюс (после комиссии) минимум в 4 из 5 хронологических групп контрактов,
и калибровка по суррогатам (перемешанные дни), чтобы отличить находку от случайного перебора."""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from study import config_grid, evaluate, group_stats, load_contracts, make_surrogate, split_groups

REPORTS = Path(__file__).resolve().parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)
N_SURR = int(sys.argv[1]) if len(sys.argv) > 1 else 100

CTS = load_contracts()
GRID = config_grid()
GROUPS = split_groups(CTS)
IDX = [np.array([i for i, c in enumerate(CTS) if c.secid in g]) for g in GROUPS]


IS_STORAGE = np.array([g[4] != "none" for g in GRID])
BASE_IDX = {(g[0], g[1], g[2]): j for j, g in enumerate(GRID) if g[4] == "none" and g[3] == 0.0}
PAIRS = {b: [(j, BASE_IDX[(g[0], g[1], g[2])]) for j, g in enumerate(GRID) if g[4] == b and g[0] != "storage"]
         for b in ("dev4", "level")}


def paired_gain(mean_net: np.ndarray) -> dict:
    """Средний прирост net от фильтра по запасам относительно такой же стратегии без фильтра."""
    return {f"gain_{b}": float(np.mean([mean_net[j] - mean_net[k] for j, k in p])) for b, p in PAIRS.items()}


def stats_of(net: np.ndarray) -> dict:
    gs = group_stats(net, IDX)               # [5, configs]
    plus = (gs > 0).sum(axis=0)
    return {"gs": gs, "plus": plus, "mean": net.mean(axis=0)}


def surrogate_run(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    net = evaluate([make_surrogate(c, rng) for c in CTS], GRID)
    s = stats_of(net)
    return {"n4": int((s["plus"] >= 4).sum()), "n5": int((s["plus"] == 5).sum()),
            "n4_storage": int(((s["plus"] >= 4) & IS_STORAGE).sum()),
            "best_mean": float(s["mean"].max()), "best_mean_storage": float(s["mean"][IS_STORAGE].max()),
            "best_min_group": float(s["gs"].min(axis=0).max()), **paired_gain(s["mean"])}


if __name__ == "__main__":
    net = evaluate(CTS, GRID)
    s = stats_of(net)
    res = pd.DataFrame(GRID, columns=["family", "n", "step", "er", "bias"])
    for k in range(5):
        res[f"g{k+1}"] = s["gs"][k]
    res["plus_groups"] = s["plus"]
    res["net_mean"] = s["mean"]
    res["net_median"] = np.median(net, axis=0)
    res["win_contracts"] = (net > 0).mean(axis=0)
    detail = evaluate(CTS, GRID, detail=True)
    agg = detail.groupby(["family", "n", "step", "er", "bias"]).agg(
        trades=("trades", "mean"), gross=("gross", "mean"), fees=("fees", "mean"), coverage=("coverage", "mean")).reset_index()
    res = res.merge(agg, on=["family", "n", "step", "er", "bias"])
    res.to_csv(REPORTS / "search_results.csv", index=False)

    print("Группы:", [f"{g[0]}..{g[-1]}({len(g)})" for g in GROUPS])
    print(f"конфигураций: {len(GRID)}; плюс в >=4 группах: {(res.plus_groups >= 4).sum()}; в 5 группах: {(res.plus_groups == 5).sum()}")
    pd.set_option("display.width", 250)
    cols = ["family", "n", "step", "er", "bias", "g1", "g2", "g3", "g4", "g5", "plus_groups", "net_mean", "net_median", "trades", "gross", "coverage"]
    print("\nКонфигурации с плюсом в >=4 группах из 5 (по net_mean):")
    print(res[res.plus_groups >= 4].sort_values("net_mean", ascending=False)[cols].head(40).round(4).to_string(index=False))

    print(f"\nСуррогаты (перемешанные дни, {N_SURR} шт.): сколько конфигураций проходит критерий случайно")
    with Pool(4) as p:
        sur = pd.DataFrame(p.map(surrogate_run, range(1000, 1000 + N_SURR)))
    sur.to_csv(REPORTS / "surrogates.csv", index=False)
    real = {"n4": int((s["plus"] >= 4).sum()), "n5": int((s["plus"] == 5).sum()),
            "n4_storage": int(((s["plus"] >= 4) & IS_STORAGE).sum()),
            "best_mean": float(s["mean"].max()), "best_mean_storage": float(s["mean"][IS_STORAGE].max()),
            "best_min_group": float(s["gs"].min(axis=0).max()), **paired_gain(s["mean"])}
    print(sur.describe().round(4).to_string())
    print("реальные данные:", {k: round(v, 4) for k, v in real.items()})
    for k in real:
        print(f"  p-value {k}: {(sur[k] >= real[k]).mean():.3f}")
    print("\nМесто storage-конфигов: средний net семейства storage:",
          {c: round(float(res[(res.family == 'storage') & (res.bias == c)].net_mean.iloc[0]), 4) for c in ("dev4", "level")})
