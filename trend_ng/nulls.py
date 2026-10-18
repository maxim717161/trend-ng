"""Нулевые модели (суррогатные ряды) для проверки трендовости.

signflip - каждый шаг (разрыв open/prev close и внутрибаровое движение) независимо умножается на случайный знак:
           сохраняет распределение и кластеризацию волатильности, убирает любую зависимость и дрейф.
           Мартингал: у любой нетрендовой стратегии валовое математическое ожидание = 0.
daylbock - дни перемешаны блоками (study.make_surrogate): нуль для многодневной зависимости, но у перестановки
           есть эффект «моста» (сумма шагов фиксирована), см. README.
"""
from __future__ import annotations

import numpy as np

from study import Contract, make_surrogate


def make_signflip(ct: Contract, rng: np.random.Generator) -> Contract:
    prev_c = np.r_[ct.c[0], ct.c[:-1]]
    g = (ct.o / prev_c - 1) * rng.choice([-1.0, 1.0], len(ct.c))
    r = (ct.c / ct.o - 1) * rng.choice([-1.0, 1.0], len(ct.c))
    c = ct.c[0] * np.cumprod((1 + g) * (1 + r))
    o = np.r_[ct.c[0], c[:-1]] * (1 + g)
    return Contract(ct.secid, ct.expiry, ct.trade_start, ct.ts, o, c)


NULLS = {"signflip": make_signflip, "dayblock": make_surrogate}
