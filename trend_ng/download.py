"""Минутные свечи по каждому месячному контракту NGM (тикеры NR*) с MOEX ISS.

Контракты НЕ склеиваются: каждый контракт хранится отдельным файлом
data/minute/<secid>_1m.parquet (весь доступный период жизни контракта).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://iss.moex.com/iss"
ENGINE = f"{BASE}/engines/futures/markets/forts"
MONTH_CODES = "FGHJKMNQUVXZ"
DATA = Path(__file__).resolve().parent.parent / "data" / "minute"


def get(url: str, params: dict | None = None, retries: int = 6) -> dict:
    params = {"iss.meta": "off", **(params or {})}
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=40)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def candidates(years=range(2024, 2028)) -> list[str]:
    return [f"NR{m}{y % 10}" for y in years for m in MONTH_CODES]


def minute_borders(secid: str) -> tuple[str, str] | None:
    d = get(f"{ENGINE}/securities/{secid}/candleborders.json")["borders"]
    for begin, end, interval, _ in d["data"]:
        if interval == 1:
            return begin, end
    return None


def download(secid: str) -> dict | None:
    b = minute_borders(secid)
    if b is None:
        return None
    begin, end = b
    frames, start = [], 0
    while True:
        j = get(f"{ENGINE}/securities/{secid}/candles.json",
                {"interval": 1, "from": begin[:10], "till": end[:10] + " 23:59:59", "start": start})["candles"]
        if not j["data"]:
            break
        frames.append(pd.DataFrame(j["data"], columns=j["columns"]))
        start += len(j["data"])
    df = pd.concat(frames, ignore_index=True)
    df["begin"] = pd.to_datetime(df["begin"])
    df["end"] = pd.to_datetime(df["end"])
    df = df.drop_duplicates("begin").sort_values("begin").reset_index(drop=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA / f"{secid}_1m.parquet")
    return {"secid": secid, "first": df["begin"].iloc[0], "last": df["begin"].iloc[-1], "bars": len(df)}


def main() -> None:
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = [r for r in ex.map(download, candidates()) if r]
    out = pd.DataFrame(res).sort_values("first")
    out.to_csv(DATA / "contracts.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
