"""Реакция NGM на отчёт EIA по запасам и торговля вокруг отчёта (контракты не склеиваются)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import eia
from events import HORIZONS, build_events, load_all

REPORTS = Path(__file__).resolve().parent.parent / "reports"
FEE = 0.0005
rng = np.random.default_rng(11)
pd.set_option("display.width", 250)

ev = build_events()
ev.to_csv(REPORTS / "eia_events.csv", index=False)
bars = load_all()
print(f"событий: {len(ev)} ({ev.t0.min():%Y-%m-%d} .. {ev.t0.max():%Y-%m-%d})")

# 1. проверка времени публикации: средний |доходность| минутной свечи вокруг отчёта и в те же минуты в контрольные дни
print("\n== 1. Проверка привязки по времени: средний |ret| 1-минутных свечей, % ==")
rows = []
for off in (-5, -2, -1, 0, 1, 2, 3, 5, 10, 30):
    a, b = [], []
    for _, e in ev.iterrows():
        c = bars[e["secid"]]["close"]
        for dd, lst in ((0, a), (-1, b), (-2, b), (-3, b), (1, b)):
            t = e["t0"] + pd.Timedelta(days=dd, minutes=off)
            if t in c.index and (t - pd.Timedelta(minutes=1)) in c.index:
                lst.append(abs(np.log(c[t] / c[t - pd.Timedelta(minutes=1)])))
    rows.append({"минут от публикации": off, "день отчёта": np.mean(a) * 100, "контроль (те же минуты в Пн-Ср, Пт)": np.mean(b) * 100,
                 "отношение": np.mean(a) / np.mean(b)})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# 2. реакция против сюрприза
print("\n== 2. Связь реакции цены с признаками отчёта (ожидаем отрицательную: меньше инъекция -> цена вверх) ==")
def perm_p(x: np.ndarray, y: np.ndarray, n: int = 5000) -> tuple[float, float]:
    r = pd.Series(x).corr(pd.Series(y), method="spearman")
    null = np.array([pd.Series(rng.permutation(x)).corr(pd.Series(y), method="spearman") for _ in range(n)])
    return r, (np.abs(null) >= abs(r)).mean()
rows = []
for feat in ("surprise", "dev_change", "change"):
    d = ev.dropna(subset=[feat])
    for h in HORIZONS:
        dd = d.dropna(subset=[f"R{h}"])
        r, p = perm_p(dd[feat].to_numpy(), dd[f"R{h}"].to_numpy(), 2000)
        slope = np.polyfit(dd[feat], dd[f"R{h}"], 1)[0] * 10_000 * 10   # б.п. на каждые 10 Bcf
        rows.append({"признак": feat, "горизонт, мин": h, "n": len(dd), "Spearman": r, "p-value": p, "б.п. цены на +10 Bcf": slope})
corr = pd.DataFrame(rows)
print(corr.round(3).to_string(index=False))
corr.to_csv(REPORTS / "eia_reaction_corr.csv", index=False)
print("\nСредняя абсолютная реакция |R_h|, %:", {h: round(ev[f'R{h}'].abs().mean() * 100, 2) for h in HORIZONS})

# 3. торговля: матрица незнаковых доходностей [событие, (вход d, удержание H)]
DELAYS, HOLDS = (1, 5, 15), (15, 30, 60, 120, 240)

def price_at(sec: str, t: pd.Timestamp) -> float:
    c = bars[sec]["close"]
    seg = c.loc[: t - pd.Timedelta(minutes=1)]
    return seg.iloc[-1]

ret = {}
for d in DELAYS:
    for H in HOLDS:
        vals = []
        for _, e in ev.iterrows():
            c = bars[e["secid"]]["close"]
            t_in, t_out = e["t0"] + pd.Timedelta(minutes=d), e["t0"] + pd.Timedelta(minutes=d + H)
            if c.index[-1] < t_out:
                vals.append(np.nan)
                continue
            vals.append(price_at(e["secid"], t_out) / price_at(e["secid"], t_in) - 1)
        ret[(d, H)] = np.array(vals)
React = {d: ev[f"R{d}"].to_numpy() for d in DELAYS if f"R{d}" in ev}
React = {1: ev["R1"].to_numpy(), 5: ev["R5"].to_numpy(), 15: ev["R15"].to_numpy()}

def positions(mode: str, d: int, sur: np.ndarray, dev: np.ndarray) -> np.ndarray:
    if mode == "surprise":
        return -np.sign(sur)
    if mode == "dev_change":
        return -np.sign(dev)
    if mode == "momentum":
        return np.sign(React[d])
    if mode == "fade":
        return -np.sign(React[d])
    raise ValueError

MODES = ("surprise", "dev_change", "momentum", "fade")
cfgs = [(m, d, H) for m in MODES for d in DELAYS for H in HOLDS]
valid = np.all([~np.isnan(ret[(d, H)]) for d in DELAYS for H in HOLDS], axis=0) & ev["surprise"].notna().to_numpy() & ev["dev_change"].notna().to_numpy() \
        & np.isfinite(ev["R1"].to_numpy()) & np.isfinite(ev["R5"].to_numpy()) & np.isfinite(ev["R15"].to_numpy())
idx = np.flatnonzero(valid)
groups = np.array_split(idx, 5)
print(f"\n== 3. Торговля вокруг отчёта: {len(idx)} событий, 5 групп по {[len(g) for g in groups]} ==")

def evaluate(sur: np.ndarray, dev: np.ndarray, signs: np.ndarray | None = None):
    out = np.zeros((len(idx), len(cfgs)))
    for j, (m, d, H) in enumerate(cfgs):
        pos = positions(m, d, sur, dev)[idx]
        if signs is not None and m in ("momentum", "fade"):
            pos = pos * signs
        out[:, j] = pos * ret[(d, H)][idx] - 2 * FEE * (pos != 0)
    return out

sur, dev = ev["surprise"].to_numpy(), ev["dev_change"].to_numpy()
net = evaluate(sur, dev)
gm = np.vstack([net[np.searchsorted(idx, g)].mean(axis=0) for g in groups])
res = pd.DataFrame(cfgs, columns=["режим", "вход через, мин", "удержание, мин"])
for k in range(5):
    res[f"G{k+1}, %"] = gm[k] * 100
res["групп с плюсом"] = (gm > 0).sum(axis=0)
res["среднее net, %"] = net.mean(axis=0) * 100
res["доля плюсовых сделок"] = (net > 0).mean(axis=0)
res["t"] = net.mean(axis=0) / (net.std(axis=0, ddof=1) / np.sqrt(len(idx)))
res["gross, %"] = (net.mean(axis=0) + 2 * FEE) * 100
res.to_csv(REPORTS / "eia_event_strategies.csv", index=False)
print(res.sort_values("среднее net, %", ascending=False).head(12).round(3).to_string(index=False))
print("\nСредний net по режимам, %:", res.groupby("режим")["среднее net, %"].mean().round(3).to_dict())
print("Конфигураций с плюсом в >=4 группах:", int((res["групп с плюсом"] >= 4).sum()), "из", len(res))
print(res[res["групп с плюсом"] >= 4].round(3).to_string(index=False))

# 4. перестановочный тест: перемешиваем признак запасов между событиями / знаки реакции
print("\n== 4. Перестановочный тест (5000 перестановок) ==")
n_perm = 5000
null_best, null_n4 = np.zeros(n_perm), np.zeros(n_perm)
for i in range(n_perm):
    perm = rng.permutation(len(ev))
    signs = rng.choice([-1.0, 1.0], size=len(idx))
    nn = evaluate(sur[perm], dev[perm], signs)
    g = np.vstack([nn[np.searchsorted(idx, gg)].mean(axis=0) for gg in groups])
    null_best[i] = nn.mean(axis=0).max()
    null_n4[i] = ((g > 0).sum(axis=0) >= 4).sum()
real_best = net.mean(axis=0).max()
real_n4 = int((res["групп с плюсом"] >= 4).sum())
print(f"лучший средний net: реальный {real_best*100:.3f}%, нуль: среднее {null_best.mean()*100:.3f}%, 95-й перцентиль {np.quantile(null_best, .95)*100:.3f}%, p = {(null_best >= real_best).mean():.4f}")
print(f"конфигураций с >=4 группами: реальные {real_n4}, нуль: среднее {null_n4.mean():.1f}, p = {(null_n4 >= real_n4).mean():.4f}")
for m in MODES:
    cols = [j for j, c in enumerate(cfgs) if c[0] == m]
    print(f"  режим {m}: лучший средний net {net[:, cols].mean(axis=0).max()*100:.3f}%")
