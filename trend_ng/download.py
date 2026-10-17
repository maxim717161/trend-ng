"""Загрузка истории фьючерса NGM (Natural Gas mini, MOEX FORTS) через ISS.

Скачиваются все вечные месячные контракты NR<месяц><год>:
  - часовые свечи (interval=60)
  - дневная история (OI, объём, расчётная цена)
Результат кладётся в data/raw/*.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://iss.moex.com/iss"
ENGINE = f"{BASE}/engines/futures/markets/forts"
MONTH_CODES = "FGHJKMNQUVXZ"
ASSET_PREFIX = "NR"  # NGM -> тикер NR*
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

session = requests.Session()


def get(url: str, params: dict | None = None, retries: int = 5) -> dict:
    params = {"iss.meta": "off", **(params or {})}
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def paged(url: str, block: str, params: dict | None = None, page: int = 500) -> pd.DataFrame:
    frames, start = [], 0
    while True:
        j = get(url, {**(params or {}), "start": start})
        d = j[block]
        df = pd.DataFrame(d["data"], columns=d["columns"])
        if df.empty:
            break
        frames.append(df)
        start += len(df)
        if len(df) < 1:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def candidate_secids(years=range(2019, 2028)) -> list[tuple[str, int]]:
    return [(f"{ASSET_PREFIX}{m}{y % 10}", y) for y in years for m in MONTH_CODES]


def borders(secid: str) -> pd.DataFrame:
    j = get(f"{ENGINE}/securities/{secid}/candleborders.json")
    d = j["borders"]
    return pd.DataFrame(d["data"], columns=d["columns"])


def download_contract(secid: str) -> dict | None:
    b = borders(secid)
    if b.empty:
        return None
    h = b[b.interval == 60].iloc[0]
    begin, end = h["begin"][:10], h["end"][:10]
    candles = paged(
        f"{ENGINE}/securities/{secid}/candles.json",
        "candles",
        {"interval": 60, "from": begin, "till": end},
    )
    hist = paged(
        f"{BASE}/history/engines/futures/markets/forts/securities/{secid}.json",
        "history",
        {"from": begin, "till": end},
    )
    if candles.empty:
        return None
    candles["begin"] = pd.to_datetime(candles["begin"])
    candles["end"] = pd.to_datetime(candles["end"])
    candles["secid"] = secid
    RAW.mkdir(parents=True, exist_ok=True)
    candles.to_parquet(RAW / f"{secid}_60m.parquet")
    if not hist.empty:
        hist.to_parquet(RAW / f"{secid}_daily.parquet")
    return {"secid": secid, "first": begin, "last": end, "bars": len(candles), "days": len(hist)}


def main() -> None:
    out = []
    seen = set()
    for secid, year in candidate_secids():
        # единичная цифра года неоднозначна по десятилетиям: берём первый существующий
        if secid in seen:
            continue
        info = download_contract(secid)
        if info:
            seen.add(secid)
            out.append(info)
            print(info, flush=True)
    pd.DataFrame(out).to_csv(RAW / "contracts.csv", index=False)


if __name__ == "__main__":
    main()
