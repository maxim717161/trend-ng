"""Поиск трендовой стратегии на NGM (минутки, контракты не склеиваются).

Для каждого завершённого контракта отдельно:
  [max(листинг, экспирация-WARM_MONTHS), экспирация-1м)  - прогрев, позиции нет
  [экспирация-1м, экспирация]                            - торговля; на последней минуте позиция закрывается

Параметры стратегии:
  family  - donchian / ema_cross / momentum
  n       - размер окна в минутах (<= MAX_WINDOW)
  step    - шаг принятия решений в минутах (стратегия считается на барах step минут)
  er      - порог efficiency ratio окна: при ER ниже порога позиция 0 (0 = фильтр выключен)
Сигнал по закрытию бара исполняется по open следующего бара. Комиссия 0.05% за сторону.
Окно включается, когда накоплено достаточно баров (иначе позиция 0, см. coverage).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "minute"
FEE = 0.0005
WARM_MONTHS = 4
MAX_WINDOW = 10_000
WINDOWS = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 1920, 2880, 4320, 5760, 7200, 10000]
STEPS = [1, 5, 15, 30, 60, 120, 240]
FAMILIES = ("donchian", "ema_cross", "momentum")
ER_LEVELS = (0.0, 0.2)
FAST_DIV = 4
MIN_BARS_PER_WINDOW = 4  # окно должно содержать хотя бы столько баров решения


@dataclass
class Contract:
    secid: str
    expiry: pd.Timestamp
    trade_start: pd.Timestamp
    ts: pd.DatetimeIndex
    o: np.ndarray
    c: np.ndarray

    @property
    def in_trade(self) -> np.ndarray:
        return self.ts >= self.trade_start


def load_contracts(warm_months: int = WARM_MONTHS) -> list[Contract]:
    raw = {p.name.split("_")[0]: pd.read_parquet(p).set_index("begin") for p in sorted(DATA.glob("*_1m.parquet"))}
    last_ts = max(df.index.max() for df in raw.values())
    out = []
    for secid, df in raw.items():
        if df.index.max() >= last_ts - pd.Timedelta(days=1):
            continue  # контракт ещё торгуется
        expiry = df.index.max().normalize()
        trade_start = expiry - pd.DateOffset(months=1)
        df = df.loc[expiry - pd.DateOffset(months=warm_months):]
        out.append(Contract(secid, expiry, trade_start, df.index, df["open"].to_numpy(), df["close"].to_numpy()))
    return sorted(out, key=lambda c: c.expiry)


def split_groups(contracts: list[Contract], k: int = 5) -> list[list[str]]:
    """Хронологические группы по дате экспирации."""
    return [[c.secid for c in part] for part in np.array_split(np.array(contracts, dtype=object), k)]


def resample(ct: Contract, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Бары по step минут: open, close, признак торговой части."""
    if step == 1:
        return ct.o, ct.c, ct.in_trade
    key = ct.ts.floor(f"{step}min").asi8
    change = np.r_[True, key[1:] != key[:-1]]
    first = np.flatnonzero(change)
    last = np.r_[first[1:] - 1, len(key) - 1]
    return ct.o[first], ct.c[last], ct.ts[first] >= ct.trade_start


def make_surrogate(ct: Contract, rng: np.random.Generator) -> Contract:
    """Перемешивание торговых дней блоками: сохраняет внутридневную структуру,
    разрушает зависимость между днями (нулевая гипотеза: нет многодневного тренда)."""
    prev_c = np.r_[ct.c[0], ct.c[:-1]]
    gap = ct.o / prev_c - 1
    intra = ct.c / ct.o - 1
    day = ct.ts.normalize().asi8
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], len(day)]
    order = rng.permutation(len(starts))
    idx = np.concatenate([np.arange(starts[i], ends[i]) for i in order])
    g, r = gap[idx], intra[idx]
    c = ct.c[0] * np.cumprod((1 + g) * (1 + r))
    o = np.r_[ct.c[0], c[:-1]] * (1 + g)
    return Contract(ct.secid, ct.expiry, ct.trade_start, ct.ts, o, c)


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def raw_signal(family: str, c: np.ndarray, m: int) -> np.ndarray:
    s = pd.Series(c)
    if family == "donchian":
        hi, lo = s.rolling(m).max().shift(1), s.rolling(m).min().shift(1)
        sig = pd.Series(np.nan, index=s.index)
        sig[s > hi] = 1.0
        sig[s < lo] = -1.0
        return sig.ffill().fillna(0.0).to_numpy()
    if family == "ema_cross":
        return np.sign(_ema(c, max(2, m // FAST_DIV)) - _ema(c, m))
    if family == "momentum":
        return np.nan_to_num(np.sign(c - s.shift(m).to_numpy()))
    raise ValueError(family)


def efficiency_ratio(c: np.ndarray, m: int) -> np.ndarray:
    s = pd.Series(c)
    path = s.diff().abs().rolling(m).sum()
    return ((s - s.shift(m)).abs() / path).to_numpy()


def run(o: np.ndarray, c: np.ndarray, in_trade: np.ndarray, family: str, m: int, er: float, fee: float = FEE) -> dict:
    sig = raw_signal(family, c, m)
    if er > 0:
        sig = np.where(efficiency_ratio(c, m) >= er, sig, 0.0)
    ready = np.arange(len(c)) + 1 >= (m if family == "ema_cross" else m + 1)
    sig = np.where(in_trade & ready, sig, 0.0)
    pos_new = np.r_[0.0, sig[:-1]]
    pos_new[-1] = 0.0
    pos_old = np.r_[0.0, pos_new[:-1]]
    prev_c = np.r_[c[0], c[:-1]]
    gross = pos_old * (o / prev_c - 1) + pos_new * (c / o - 1)
    turnover = np.abs(pos_new - pos_old)
    mk = in_trade
    return {"gross": gross[mk].sum(), "fees": fee * turnover[mk].sum(),
            "net": gross[mk].sum() - fee * turnover[mk].sum(),
            "trades": int(((turnover > 0) & mk).sum()), "coverage": float(ready[mk].mean())}


def config_grid() -> list[tuple[str, int, int, float]]:
    grid = []
    for f in FAMILIES:
        for step in STEPS:
            for n in WINDOWS:
                if n > MAX_WINDOW or n // step < MIN_BARS_PER_WINDOW:
                    continue
                for er in ER_LEVELS:
                    grid.append((f, n, step, er))
    return grid


def evaluate(contracts: list[Contract], grid=None, fee: float = FEE, detail: bool = False):
    """Матрицы результатов [контракт, конфиг]. detail=True -> DataFrame со всеми метриками."""
    grid = grid or config_grid()
    rows = []
    net = np.zeros((len(contracts), len(grid)))
    for i, ct in enumerate(contracts):
        cache = {s: resample(ct, s) for s in sorted({g[2] for g in grid})}
        for j, (f, n, step, er) in enumerate(grid):
            o, c, it = cache[step]
            r = run(o, c, it, f, n // step, er, fee)
            net[i, j] = r["net"]
            if detail:
                rows.append({"secid": ct.secid, "family": f, "n": n, "step": step, "er": er, **r})
    return pd.DataFrame(rows) if detail else net


def group_stats(net: np.ndarray, groups_idx: list[np.ndarray]) -> np.ndarray:
    """Средний net по каждой группе: [группа, конфиг]."""
    return np.vstack([net[ix].mean(axis=0) for ix in groups_idx])
