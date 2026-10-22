"""Бэктест боевого правила на исторических минутках (data/minute, см. trend_ng/download.py).

Правило контракта как в боевом коде: торгуется контракт с ближайшей экспирацией, до которой больше GUARD_DAYS дней;
за GUARD_DAYS до экспирации позиция закрывается, торговля переходит на следующий контракт (его сигнал считается по его собственной истории).
Решение на закрытии часового бара, исполнение по open следующего бара, комиссия 0.05% за сторону + необязательное проскальзывание.

Запуск: python3 -m trader.backtest   (из корня репозитория)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trend_ng"))
from study import load_contracts, resample  # noqa: E402

from trader.strategy import decide  # noqa: E402

FEE = 0.0005
GUARD_DAYS = 3
STEP = 60


def run_contract(ct, start, end, slip: float = 0.0) -> dict:
    o, c, _, first = resample(ct, STEP)
    ts = ct.ts[first]
    win = (ts >= start) & (ts < end)
    pos_prev, target = 0, 0
    gross = turnover = 0.0
    trades = 0
    idx = np.flatnonzero(win)
    for n, i in enumerate(idx):
        last = n == len(idx) - 1
        gap = (o[i] / c[i - 1] - 1) if i > 0 else 0.0
        intra = c[i] / o[i] - 1
        pos_new = 0 if last else target
        gross += pos_prev * gap + pos_new * intra
        d = abs(pos_new - pos_prev)
        turnover += d
        trades += d > 0
        pos_prev = pos_new
        target, _ = decide(c[: i + 1], target)
    return {"gross": gross, "net": gross - (FEE + slip) * turnover, "trades": int(trades), "turnover": turnover}


def main() -> None:
    cts = load_contracts()
    rows = []
    prev_exp = None
    for ct in cts:
        end = ct.expiry - pd.Timedelta(days=GUARD_DAYS)
        start = (prev_exp - pd.Timedelta(days=GUARD_DAYS)) if prev_exp is not None else ct.trade_start
        prev_exp = ct.expiry
        row = {"secid": ct.secid, "start": start.date(), "end": end.date()}
        for slip in (0.0, 0.0002, 0.0005):
            r = run_contract(ct, start, end, slip)
            row[f"net_slip{int(slip * 1e4)}bp"] = r["net"] * 100
            if slip == 0:
                row.update({"gross": r["gross"] * 100, "trades": r["trades"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    cols = ["gross", "net_slip0bp", "net_slip2bp", "net_slip5bp"]
    print("\nсреднее на контракт-месяц, % номинала:")
    print(df[cols].mean().round(2).to_string())
    print("доля плюсовых контракт-месяцев:", {c: round(float((df[c] > 0).mean()), 2) for c in cols})
    print("медиана:", df[cols].median().round(2).to_dict(), "| сделок в месяц в среднем:", round(df["trades"].mean(), 1))
    print("без NRF6:", df[df.secid != "NRF6"][cols].mean().round(2).to_dict())
    groups = np.array_split(np.arange(len(df)), 5)
    print("net без проскальзывания по 5 группам:", [round(df["net_slip0bp"].iloc[g].mean(), 2) for g in groups])
    print("стандартное отклонение месячного net, %:", round(df["net_slip0bp"].std(), 1), "| худший месяц:", round(df["net_slip0bp"].min(), 1))
    df.to_csv(Path(__file__).resolve().parent.parent / "reports" / "trader_backtest.csv", index=False)


if __name__ == "__main__":
    main()
