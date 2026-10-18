"""Поиск контртрендовой (возврат к среднему) стратегии: z-score против скользящего среднего.

Вход против отклонения: z > k -> шорт, z < -k -> лонг, z = (close - SMA_n) / STD_n.
Выход при возврате к среднему (z пересекает 0); опционально стоп при |z| > 4 (после стопа вход
блокируется, пока |z| не вернётся ниже k). Окно n <= 10000 минут, шаг решений 5-240 минут.
Критерий как в основном исследовании: плюс (после комиссии 0.05%) минимум в 4 из 5 хронологических групп контрактов.
Нуль: случайное блуждание с той же волатильностью (случайные знаки шагов), суррогаты.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from study import FEE, Contract, load_contracts, resample, split_groups

REPORTS = Path(__file__).resolve().parent.parent / "reports"
N_SURR = int(sys.argv[1]) if len(sys.argv) > 1 else 100
STEPS = [5, 15, 30, 60, 120, 240]
WINDOWS = [30, 60, 120, 240, 480, 960, 1440, 2880, 5760, 10000]
KS = [1.0, 1.5, 2.0, 2.5, 3.0]
STOPS = [0.0, 4.0]


@njit(cache=True)
def zrev_sig(z, k, stop, ready):
    n = len(z)
    sig = np.zeros(n)
    pos = 0.0
    blocked = False
    for i in range(n):
        zi = z[i]
        if (not ready[i]) or np.isnan(zi):
            pos = 0.0
            sig[i] = 0.0
            continue
        if blocked and abs(zi) < k:
            blocked = False
        if pos == 0.0:
            if not blocked:
                if zi > k:
                    pos = -1.0
                elif zi < -k:
                    pos = 1.0
        elif pos < 0.0:
            if zi <= 0.0:
                pos = 0.0
            elif stop > 0.0 and zi > stop:
                pos = 0.0
                blocked = True
        else:
            if zi >= 0.0:
                pos = 0.0
            elif stop > 0.0 and zi < -stop:
                pos = 0.0
                blocked = True
        sig[i] = pos
    return sig


def run(o, c, in_trade, m, k, stop, fee=FEE):
    s = pd.Series(c)
    mu, sd = s.rolling(m).mean(), s.rolling(m).std(ddof=0)
    z = ((s - mu) / sd).to_numpy()
    ready = np.arange(len(c)) + 1 >= m
    sig = zrev_sig(z, k, stop, ready)
    sig = np.where(in_trade, sig, 0.0)
    pos_new = np.r_[0.0, sig[:-1]]
    pos_new[-1] = 0.0
    pos_old = np.r_[0.0, pos_new[:-1]]
    prev_c = np.r_[c[0], c[:-1]]
    gross = pos_old * (o / prev_c - 1) + pos_new * (c / o - 1)
    turn = np.abs(pos_new - pos_old)
    g, f = gross[in_trade].sum(), fee * turn[in_trade].sum()
    return g, f, int(((turn > 0) & in_trade).sum())


def grid():
    return [(n, st, k, sp) for st in STEPS for n in WINDOWS if n // st >= 6 for k in KS for sp in STOPS]


def make_rw(ct: Contract, rng: np.random.Generator) -> Contract:
    """Нуль: каждый шаг (разрыв open/prev close и внутрибаровое движение) независимо умножается на случайный знак.
    Сохраняет распределение и кластеризацию волатильности, убирает любую зависимость и дрейф."""
    prev_c = np.r_[ct.c[0], ct.c[:-1]]
    g = (ct.o / prev_c - 1) * rng.choice([-1.0, 1.0], len(ct.c))
    r = (ct.c / ct.o - 1) * rng.choice([-1.0, 1.0], len(ct.c))
    c = ct.c[0] * np.cumprod((1 + g) * (1 + r))
    o = np.r_[ct.c[0], c[:-1]] * (1 + g)
    return Contract(ct.secid, ct.expiry, ct.trade_start, ct.ts, o, c)


def evaluate(cts, G, detail=False):
    net = np.zeros((len(cts), len(G)))
    gross = np.zeros_like(net)
    trades = np.zeros_like(net)
    for i, ct in enumerate(cts):
        cache = {s: resample(ct, s) for s in STEPS}
        for j, (n, st, k, sp) in enumerate(G):
            o, c, it, _ = cache[st]
            g, f, t = run(o, c, it, n // st, k, sp)
            net[i, j], gross[i, j], trades[i, j] = g - f, g, t
    return (net, gross, trades) if detail else net


CTS = load_contracts()
G = grid()
GROUPS = split_groups(CTS)
IDX = [np.array([i for i, c in enumerate(CTS) if c.secid in g]) for g in GROUPS]


def stat(net):
    gs = np.vstack([net[ix].mean(axis=0) for ix in IDX])
    plus = (gs > 0).sum(axis=0)
    return gs, plus


def surrogate(seed):
    rng = np.random.default_rng(seed)
    net = evaluate([make_rw(c, rng) for c in CTS], G)
    gs, plus = stat(net)
    return {"n4": int((plus >= 4).sum()), "n5": int((plus == 5).sum()), "best_mean": float(net.mean(0).max()),
            "mean_all": float(net.mean()), "share_pos": float((net.mean(0) > 0).mean())}


if __name__ == "__main__":
    net, gross, trades = evaluate(CTS, G, detail=True)
    gs, plus = stat(net)
    res = pd.DataFrame(G, columns=["n", "step", "k", "stop"])
    for q in range(5):
        res[f"g{q+1}"] = gs[q]
    res["plus_groups"], res["net_mean"] = plus, net.mean(0)
    res["gross"], res["trades"], res["win_contracts"] = gross.mean(0), trades.mean(0), (net > 0).mean(0)
    res.to_csv(REPORTS / "counter_search_results.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"конфигураций {len(G)}; групп: {[len(g) for g in GROUPS]}")
    print(f"средний net по всем конфигурациям {net.mean()*100:.2f}%, средний gross {gross.mean()*100:.2f}%, "
          f"доля конфигураций с net_mean>0: {(net.mean(0) > 0).mean():.1%}")
    print(f"плюс в >=4 группах: {(plus >= 4).sum()}, в 5 группах: {(plus == 5).sum()}")
    print("\nТоп-15 по net_mean:")
    print(res.sort_values("net_mean", ascending=False).head(15).round(4).to_string(index=False))
    print("\nПо шагу/окну: средний gross и net, %")
    print((res.groupby("n")[["gross", "net_mean", "trades"]].mean() * [100, 100, 1]).round(2).to_string())
    print("\nПрошедшие критерий >=4 из 5:")
    print(res[res.plus_groups >= 4].sort_values("net_mean", ascending=False).round(4).to_string(index=False))
    with Pool(4) as p:
        sur = pd.DataFrame(p.map(surrogate, range(2000, 2000 + N_SURR)))
    sur.to_csv(REPORTS / "counter_surrogates.csv", index=False)
    print(f"\nСуррогаты (случайное блуждание, {N_SURR}):")
    print(sur.describe().round(4).to_string())
    real = {"n4": int((plus >= 4).sum()), "n5": int((plus == 5).sum()), "best_mean": float(net.mean(0).max()),
            "mean_all": float(net.mean()), "share_pos": float((net.mean(0) > 0).mean())}
    print("реальные:", {k: round(v, 4) for k, v in real.items()})
    for k, v in real.items():
        print(f"  p-value {k}: {(sur[k] >= v).mean():.3f}")
