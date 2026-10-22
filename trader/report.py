"""Разбор логов из бакета (или локальной папки) через месяц.

python -m trader.report [папка_с_логами_или_пусто_для_S3]  ->  печатает сводку и пишет report_events.csv, report_fills.csv
Метрики: решения и сделки, исполнение лимитных заявок (доля пассивных, время до исполнения, проскальзывание к mid на момент решения),
замкнутые сделки в % от цены входа, комиссия 0.05% за сторону (оценка), ошибки и остановки.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from trader.config import Config
from trader.storage import LocalStorage, S3Storage

FEE = 0.0005


def load_events(backend, prefix: str) -> pd.DataFrame:
    rows = []
    for key in backend.list(prefix + "logs/"):
        for line in backend.get(key).decode().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def round_trips(fills: pd.DataFrame) -> pd.DataFrame:
    pos, entry, entry_ts, out = 0, 0.0, None, []
    for r in fills.itertuples():
        for _ in range(int(r.lots)):
            if pos == 0:
                pos, entry, entry_ts = r.side, r.price, r.ts
            elif r.side == pos:
                continue
            else:
                out.append({"вход": entry_ts, "выход": r.ts, "направление": pos, "цена входа": entry, "цена выхода": r.price,
                            "gross, %": pos * (r.price - entry) / entry * 100, "net (оценка), %": pos * (r.price - entry) / entry * 100 - 2 * FEE * 100})
                pos = 0
    return pd.DataFrame(out)


def main() -> None:
    cfg = Config.from_env()
    if len(sys.argv) > 1:
        backend, prefix = LocalStorage(sys.argv[1]), cfg.s3_prefix
    else:
        backend, prefix = S3Storage(cfg.s3_bucket, cfg.s3_endpoint), cfg.s3_prefix
    ev = load_events(backend, prefix)
    if ev.empty:
        print("логов нет")
        return
    ev.to_csv("report_events.csv", index=False)
    print("период:", ev.ts.min(), "->", ev.ts.max(), "| режим:", ev["mode"].unique().tolist())
    print("события по типам:", ev["type"].value_counts().to_dict())
    for t in ("error", "exception", "safety_halt", "equity_floor", "position_mismatch", "wide_spread", "order_not_found", "order_lost_unplaced"):
        n = int((ev.type == t).sum())
        if n:
            print(f"  ВНИМАНИЕ {t}: {n}")
            print(ev[ev.type == t][["ts", *[c for c in ("error", "reason") if c in ev.columns]]].tail(5).to_string(index=False))
    dec = ev[ev.type == "decision"]
    if len(dec):
        ch = dec[dec.target != dec.prev_target]
        print(f"решений {len(dec)}, смен позиции {len(ch)}")
    fills = ev[ev.type == "fill"].copy()
    if fills.empty:
        print("исполнений нет")
        return
    fills.to_csv("report_fills.csv", index=False)
    print(f"\nисполнений: {len(fills)}; пассивных: {(fills.phase == 'passive').mean():.0%}; лотов: {int(fills.lots.sum())}")
    sl = fills.slippage_bps.dropna()
    if len(sl):
        print(f"проскальзывание к mid на момент решения, б.п. (плюс = дороже): среднее {sl.mean():+.1f}, медиана {sl.median():+.1f}, макс {sl.max():+.1f}")
        print(fills.groupby("phase").slippage_bps.agg(["count", "mean", "median"]).round(1).to_string())
    t = fills.seconds_since_decision.dropna()
    if len(t):
        print(f"время от решения до исполнения, мин: среднее {t.mean()/60:.1f}, медиана {t.median()/60:.1f}, макс {t.max()/60:.1f}")
    rt = round_trips(fills)
    if len(rt):
        print("\nзамкнутые сделки:")
        print(rt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"сумма gross {rt['gross, %'].sum():+.2f}%, net (оценка комиссии 0.05% за сторону) {rt['net (оценка), %'].sum():+.2f}%")
    else:
        print("замкнутых сделок пока нет")


if __name__ == "__main__":
    main()
