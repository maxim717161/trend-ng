from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from windows import load, scan, pick_non_overlapping, WINDOW_DAYS, WARMUP_DAYS

OUT = Path(__file__).resolve().parent.parent / "reports"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

df = load()
w = scan(df)
w.to_csv(OUT / "windows_scan.csv", index=False)
print("окон:", len(w))
print(w[["ER", "R2", "move", "net_median", "net_mean", "share_profitable", "trades_mean", "fees_mean"]].describe().round(4))

# качество фильтра: коррелирует ли трендовость окна с результатом трендовых стратегий
print("\nSpearman(score, net_median):", round(w["score"].corr(w["net_median"], method="spearman"), 3))
print("Spearman(ER, net_median):   ", round(w["ER"].corr(w["net_median"], method="spearman"), 3))
print("Spearman(|move|, net_median):", round(w["move"].abs().corr(w["net_median"], method="spearman"), 3))
pp = w.dropna(subset=["pre_ER"])
print("\nПерсистентность (есть ли предсказательная сила у месяца прогрева):")
print("Spearman(pre_ER, ER)      :", round(pp["pre_ER"].corr(pp["ER"], method="spearman"), 3))
print("Spearman(pre_ER, net_med) :", round(pp["pre_ER"].corr(pp["net_median"], method="spearman"), 3))

# неперекрывающиеся окна: календарные последовательные блоки
non = []
s = w["start"].min()
while True:
    row = w[w["start"] >= s].head(1)
    if row.empty:
        break
    non.append(row.iloc[0])
    s = row.iloc[0]["end"]
non = pd.DataFrame(non)
print("\nНеперекрывающиеся последовательные окна:")
print(non[["start", "end", "ER", "R2", "move", "dir", "net_median", "share_profitable", "trades_mean"]].round(3).to_string(index=False))
non.to_csv(OUT / "windows_sequential.csv", index=False)

trend_w = pick_non_overlapping(w, 6)
quiet_w = pick_non_overlapping(w.assign(inv=-w["score"]), 6, col="inv")
print("\nТОП трендовые окна (неперекрывающиеся):")
print(trend_w[["start", "end", "ER", "R2", "move", "dir", "net_median", "net_best", "share_profitable", "trades_mean"]].round(3).to_string(index=False))
print("\nАнтипример: самые нетрендовые окна:")
print(quiet_w[["start", "end", "ER", "R2", "move", "dir", "net_median", "share_profitable"]].round(3).to_string(index=False))
trend_w.to_csv(OUT / "windows_trend_top.csv", index=False)

# график
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df.index, df["adj_close"], lw=0.7, color="k")
ax.set_yscale("log")
for _, r in trend_w.iterrows():
    ax.axvspan(r["start"], r["end"], color="tab:green" if r["dir"] == "up" else "tab:red", alpha=0.25)
    ax.axvspan(r["start"] - pd.Timedelta(days=WARMUP_DAYS), r["start"], color="gray", alpha=0.12)
ax.set_title("NGM, непрерывный ряд (1ч): трендовые окна (зелёный=вверх, красный=вниз), серый = прогрев 1 мес.")
fig.tight_layout()
fig.savefig(OUT / "trend_windows.png", dpi=110)

fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
ax[0].plot(w["start"], w["ER"], label="ER"); ax[0].plot(w["start"], w["R2"], label="R2"); ax[0].legend()
ax[1].bar(w["start"], w["net_median"], width=5, color=np.where(w["net_median"] > 0, "g", "r"))
ax[1].set_title("медианный net PnL набора трендовых стратегий по окну (30д, шаг 7д), комиссия 0.05%/сторона")
fig.tight_layout()
fig.savefig(OUT / "window_scan.png", dpi=110)
