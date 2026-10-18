"""Связь запасов газа с доходностью торгового месяца контракта (22 наблюдения, мало мощности)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import eia
from study import bias_array, load_contracts, resample

cts = load_contracts()
st = eia.load()
rows = []
for ct in cts:
    o, c, it, first = resample(ct, 60)
    ts = ct.ts[first][it]
    d4 = bias_array(ct.ts, "dev4")[first][it]
    lv = bias_array(ct.ts, "level")[first][it]
    s0 = st[st.release_msk <= ts[0]].iloc[-1]
    s1 = st[st.release_msk <= ts[-1]].iloc[-1]
    rows.append({"secid": ct.secid, "buyhold": c[it][-1] / o[it][0] - 1, "dev_pct_start": s0.dev_pct,
                 "dev_pct_change_in_month": s1.dev_pct - s0.dev_pct, "dev_bcf_start": s0.stocks - s0.avg5_stocks,
                 "bias_dev4": d4[0], "bias_level": lv[0],
                 "sum_dev_change_in_month": st[(st.release_msk > ts[0]) & (st.release_msk <= ts[-1])].dev_change.sum()})
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
print("\nSpearman с buy&hold торгового месяца (ожидаем: дефицит/ухудшение запасов -> рост цены):")
for col in ("dev_pct_start", "dev_pct_change_in_month", "sum_dev_change_in_month"):
    print(f"  {col:28s} rho = {df[col].corr(df.buyhold, method='spearman'):+.3f}  (n={len(df)}; |rho|>0.42 нужно для p<0.05)")
for col in ("bias_dev4", "bias_level"):
    print(f"  знак {col:12s}: доля контрактов, где знак bias совпал со знаком buy&hold = {(np.sign(df[col]) == np.sign(df.buyhold)).mean():.2f}")

print("\n== Дрейф после отчёта: доходность от публикации до +1 и +5 торговых суток (тот же контракт) ==")
from events import load_all
ev = pd.read_csv("../reports/eia_events.csv", parse_dates=["t0", "week_end"])
bars = load_all()
rng = np.random.default_rng(3)
rows = []
for _, e in ev.iterrows():
    c = bars[e["secid"]]["close"]
    r = {}
    for name, days in (("R1d", 1), ("R5d", 5)):
        t = e["t0"] + pd.Timedelta(days=days)
        r[name] = np.log(c.loc[:t].iloc[-1] / e["pre"]) if c.index[-1] >= t else np.nan
    rows.append(r)
ev = pd.concat([ev, pd.DataFrame(rows)], axis=1)
for feat in ("surprise", "dev_change", "change"):
    for h in ("R1d", "R5d"):
        d = ev.dropna(subset=[feat, h])
        r = d[feat].corr(d[h], method="spearman")
        null = np.array([pd.Series(rng.permutation(d[feat].to_numpy())).corr(d[h].reset_index(drop=True), method="spearman") for _ in range(3000)])
        print(f"  {feat:11s} -> {h}: Spearman {r:+.3f}, p = {(np.abs(null) >= abs(r)).mean():.3f}, n = {len(d)}")
