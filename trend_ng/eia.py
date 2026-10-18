"""Недельные запасы газа в хранилищах США (EIA, Lower 48) и моменты публикации отчёта.

Источник: https://www.eia.gov/dnav/ng/xls/NG_STOR_WKLY_S1_W.xls (история с 2010).
Отчёт выходит по четвергам в 10:30 ET (за неделю, закончившуюся в предыдущую пятницу),
с праздничными сдвигами (расписание https://ir.eia.gov/ngs/schedule.html).
Время публикации переводится в московское (в этом же времени метки минутных свечей ISS).

Признаки на дату отчёта (известны только после публикации):
  stocks      запасы, Bcf
  change      недельное изменение, Bcf
  avg5_change среднее изменение за ту же неделю предыдущих 5 лет
  dev_change  change - avg5_change (инъекция меньше нормы = бычий сигнал)
  surprise    dev_change - rho * dev_change(предыдущей недели): прокси «сюрприза» без консенсуса аналитиков,
              rho оценивается по расширяющейся выборке (без заглядывания вперёд)
  dev_pct     запасы относительно среднего за 5 лет, %
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://www.eia.gov/dnav/ng/xls/NG_STOR_WKLY_S1_W.xls"
DATA = Path(__file__).resolve().parent.parent / "data" / "eia"

# нормальный четверг -> (фактическая дата, время ET); по расписанию EIA
HOLIDAY_SHIFTS = {
    "2024-11-28": ("2024-11-27", "12:00"),
    "2025-01-02": ("2025-01-03", "10:30"),
    "2025-01-09": ("2025-01-08", "12:00"),
    "2025-06-19": ("2025-06-18", "12:00"),
    "2025-11-13": ("2025-11-14", "10:30"),
    "2025-11-27": ("2025-11-26", "12:00"),
    "2025-12-25": ("2025-12-29", "12:00"),
    "2026-01-01": ("2025-12-31", "12:00"),
    "2026-11-12": ("2026-11-13", "10:30"),
    "2026-11-26": ("2026-11-25", "12:00"),
}


def download() -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / "NG_STOR_WKLY_S1_W.xls"
    p.write_bytes(requests.get(URL, timeout=60).content)
    return p


def release_msk(week_end: pd.Timestamp) -> pd.Timestamp:
    normal = week_end + pd.Timedelta(days=6)
    date, hhmm = HOLIDAY_SHIFTS.get(normal.strftime("%Y-%m-%d"), (normal.strftime("%Y-%m-%d"), "10:30"))
    et = pd.Timestamp(f"{date} {hhmm}").tz_localize("America/New_York")
    return et.tz_convert("Europe/Moscow").tz_localize(None)


def build() -> pd.DataFrame:
    p = DATA / "NG_STOR_WKLY_S1_W.xls"
    if not p.exists():
        download()
    raw = pd.read_excel(p, sheet_name="Data 1", header=2)
    df = raw.iloc[:, :2].copy()
    df.columns = ["week_end", "stocks"]
    df["week_end"] = pd.to_datetime(df["week_end"])
    df = df.dropna().sort_values("week_end").reset_index(drop=True)
    df["change"] = df["stocks"].diff()
    doy = df["week_end"].dt.dayofyear
    avg5_change, avg5_stocks = [], []
    for i, r in df.iterrows():
        ch, st = [], []
        for y in range(1, 6):
            tgt = r["week_end"] - pd.DateOffset(years=y)
            j = (df["week_end"] - tgt).abs().idxmin()
            if abs((df.loc[j, "week_end"] - tgt).days) <= 3:
                ch.append(df.loc[j, "change"])
                st.append(df.loc[j, "stocks"])
        avg5_change.append(np.mean(ch) if len(ch) == 5 else np.nan)
        avg5_stocks.append(np.mean(st) if len(st) == 5 else np.nan)
    df["avg5_change"], df["avg5_stocks"] = avg5_change, avg5_stocks
    df["dev_change"] = df["change"] - df["avg5_change"]
    df["dev_pct"] = (df["stocks"] / df["avg5_stocks"] - 1) * 100
    prev = df["dev_change"].shift(1)
    rho = np.full(len(df), np.nan)
    for i in range(60, len(df)):
        x, y = prev.iloc[1:i], df["dev_change"].iloc[1:i]
        m = x.notna() & y.notna()
        rho[i] = (x[m] * y[m]).sum() / (x[m] ** 2).sum()
    df["rho"] = rho
    df["surprise"] = df["dev_change"] - df["rho"] * prev
    df["release_msk"] = df["week_end"].map(release_msk)
    df.to_csv(DATA / "storage_weekly.csv", index=False)
    return df


def load() -> pd.DataFrame:
    p = DATA / "storage_weekly.csv"
    if not p.exists():
        return build()
    df = pd.read_csv(p, parse_dates=["week_end", "release_msk"])
    return df


if __name__ == "__main__":
    df = build()
    pd.set_option("display.width", 220)
    print(df.tail(8)[["week_end", "release_msk", "stocks", "change", "avg5_change", "dev_change", "surprise", "dev_pct", "rho"]].round(2).to_string(index=False))
    print("\nrho последнее:", round(df.rho.dropna().iloc[-1], 3), "; корреляция dev_change с предыдущей неделей:",
          round(df.dev_change.corr(df.dev_change.shift(1)), 3))
