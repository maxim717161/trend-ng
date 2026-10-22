"""Календарные спреды NGM: ближний контракт против следующего месяца.

Для каждого из 22 завершённых контрактов (ближний, торговый месяц - последний месяц до его экспирации) берётся следующий
по календарю контракт. Спред = «длинный ближний, короткий следующий», по 1x номинала на каждую ногу.
Синтетический ряд спреда строится из доходностей ног (разрыв open/prev close и движение внутри бара) только на минутах,
когда торговались обе ноги (иначе исполнить сделку по обеим ногам нельзя). Комиссия 0.05% за сторону на каждую ногу,
то есть 0.10% номинала одной ноги за смену позиции в спреде.

Проверяется:
  1. форма кривой по сезонам (контанго/бэквордация) и дрейф спреда за торговый месяц;
  2. статический carry: весь месяц длинный/короткий спред;
  3. динамические стратегии на спреде: тренд (donchian, ema, momentum) и возврат к среднему (z-score), 
     сетка шаг x окно, критерий «плюс в 4 группах из 5», нуль - случайные знаки шагов спреда.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from nulls import make_signflip
from study import DATA, WARM_MONTHS, Contract, load_contracts, raw_signal, resample, split_groups

REPORTS = Path(__file__).resolve().parent.parent / "reports"
N_SURR = int(sys.argv[1]) if len(sys.argv) > 1 else 300
FEE_SPREAD = 0.001
MONTHCODE = "FGHJKMNQUVXZ"
STEPS = [15, 60, 240, 720]
WINDOWS = [120, 240, 480, 960, 1440, 2880, 4320, 5760, 7200, 10000]
FAMILIES = ("momentum", "donchian", "ema_cross", "zrev1", "zrev2", "fade")
MIN_BARS = 4
GRID = [(f, h, L) for f in FAMILIES if f != "fade" for h in STEPS for L in WINDOWS if L // h >= MIN_BARS]
GRID += [("fade", h, h) for h in (15, 60, 240, 720, 1440)]
CELL_STEPS = sorted({h for _, h, _ in GRID})


def next_secid(secid: str) -> str:
    i = MONTHCODE.index(secid[2])
    year = int(secid[3])
    return "NR" + MONTHCODE[(i + 1) % 12] + str((year + (i == 11)) % 10)


def build_pairs() -> tuple[list[Contract], dict[str, float]]:
    near = load_contracts()
    spreads, info = [], {}
    for ct in near:
        nx = next_secid(ct.secid)
        raw = pd.read_parquet(DATA / f"{nx}_1m.parquet").set_index("begin")
        raw = raw[~raw.index.duplicated()]
        common = ct.ts.intersection(raw.index)
        keep = ct.ts.isin(common)
        ts = ct.ts[keep]
        on, cn = ct.o[keep], ct.c[keep]
        ox, cx = raw.loc[ts, "open"].to_numpy(), raw.loc[ts, "close"].to_numpy()
        gap = (on[1:] / cn[:-1] - 1) - (ox[1:] / cx[:-1] - 1)
        intra = (cn / on - 1) - (cx / ox - 1)
        gap = np.r_[0.0, gap]
        c = np.cumprod((1 + gap) * (1 + intra))
        o = np.r_[1.0, c[:-1]] * (1 + gap)
        sp = Contract(ct.secid, ct.expiry, ct.trade_start, ts, o, c)
        spreads.append(sp)
        tr = ts >= ct.trade_start
        lvl = np.log(cx[tr] / cn[tr]) if tr.any() else np.array([np.nan])
        info[ct.secid] = {"next": nx, "bars_trade": int(tr.sum()), "share_of_near": float(tr.sum() / (ct.ts >= ct.trade_start).sum()),
                          "level": float(np.nanmean(lvl)), "level_start": float(lvl[0]), "level_end": float(lvl[-1]),
                          "warm_bars": int((~tr).sum())}
    return spreads, info


def signal(fam: str, c: np.ndarray, k: int) -> np.ndarray:
    if fam in ("momentum", "donchian", "ema_cross"):
        return raw_signal(fam, c, k)
    if fam == "fade":
        return -np.nan_to_num(np.sign(np.r_[0.0, np.diff(c)]))
    thr = 1.0 if fam == "zrev1" else 2.0
    s = pd.Series(c)
    z = ((s - s.rolling(k).mean()) / s.rolling(k).std()).to_numpy()
    short = pd.Series(np.where(z > thr, 1.0, np.where(z <= 0, 0.0, np.nan))).ffill().fillna(0.0).to_numpy()
    long = pd.Series(np.where(z < -thr, 1.0, np.where(z >= 0, 0.0, np.nan))).ffill().fillna(0.0).to_numpy()
    return long - short


def pnl(o, c, in_trade, sig, ready, fee=FEE_SPREAD) -> tuple[float, float, int, float]:
    sig = np.where(in_trade & ready, sig, 0.0)
    pos_new = np.r_[0.0, sig[:-1]]
    pos_new[-1] = 0.0
    pos_old = np.r_[0.0, pos_new[:-1]]
    gross = pos_old * (o / np.r_[c[0], c[:-1]] - 1) + pos_new * (c / o - 1)
    turn = np.abs(pos_new - pos_old)
    g = gross[in_trade].sum()
    return g, g - fee * turn[in_trade].sum(), int(((turn > 0) & in_trade).sum()), float(turn[in_trade].sum())


def carry(sp: Contract) -> tuple[float, int]:
    tr = sp.ts >= sp.trade_start
    i0 = int(np.argmax(tr))
    return float(sp.c[-1] / sp.o[i0] - 1), int(tr.sum())


def cell_results(sp: Contract) -> np.ndarray:
    out = np.zeros((len(GRID), 4))
    cache = {h: resample(sp, h) for h in CELL_STEPS}
    for j, (f, h, L) in enumerate(GRID):
        o, c, it, _ = cache[h]
        k = L // h
        sig = signal(f, c, k)
        ready = np.arange(len(c)) + 1 >= k + 1
        out[j] = pnl(o, c, it, sig, ready)
    return out


def run_all(sps: list[Contract]) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([cell_results(s) for s in sps]), np.array([carry(s)[0] for s in sps])


SPS, INFO = build_pairs()
SEC = [s.secid for s in SPS]
GROUPS = split_groups(SPS)
IDX = [np.array([i for i, c in enumerate(SPS) if c.secid in g]) for g in GROUPS]
SEASON = np.array([{12: "зима", 1: "зима", 2: "зима", 6: "лето", 7: "лето", 8: "лето"}.get(s.expiry.month, "межсезонье") for s in SPS])


def null_run(seed: int):
    rng = np.random.default_rng(seed)
    return run_all([make_signflip(s, rng) for s in SPS])


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_columns", 30)
    info = pd.DataFrame(INFO).T
    info["expiry"] = [s.expiry.date() for s in SPS]
    info["season"] = SEASON
    real_cells, real_carry = run_all(SPS)
    info["carry_gross_%"] = real_carry * 100
    sd = [np.std(np.diff(np.log(s.c[s.ts >= s.trade_start]))) for s in SPS]
    info["spread_vol_bar"] = sd
    info.to_csv(REPORTS / "spreads_pairs.csv")
    show = info[["next", "expiry", "bars_trade", "share_of_near", "level", "level_start", "level_end", "carry_gross_%"]].copy()
    for c in ("share_of_near", "level", "level_start", "level_end", "carry_gross_%"):
        show[c] = show[c].astype(float)
    print("Пары (ближний -> следующий), торговый месяц; level = ln(следующий/ближний): >0 контанго, <0 бэквордация;")
    print("carry_gross = доходность «длинный ближний / короткий следующий» за торговый месяц (с учётом дрейфа спреда)")
    print(show.round(4).to_string())
    print(f"\nСредний уровень кривой {show['level'].mean()*100:+.2f}%, среднее carry {show['carry_gross_%'].mean():+.2f}%, "
          f"плюс у {(show['carry_gross_%'] > 0).sum()} из {len(show)}")
    print("Сезон: контрактов, средний level %, средний carry %:")
    print(info.assign(level=info["level"].astype(float), carry=info["carry_gross_%"]).groupby("season")[["level", "carry"]]
          .agg(["count", "mean"]).round(3).to_string())

    with Pool(4) as p:
        res = p.map(null_run, range(2000, 2000 + N_SURR))
    nc = np.stack([r[0] for r in res])
    ncar = np.stack([r[1] for r in res])

    gross, net, trades, turnover = (real_cells[:, :, i] for i in range(4))
    k6 = SEC.index("NRF6")
    ng, nn = nc[:, :, :, 0], nc[:, :, :, 1]
    gm, nm = gross.mean(axis=0), net.mean(axis=0)
    ngm, nnm = ng.mean(axis=1), nn.mean(axis=1)

    def groups_plus(x):
        return sum((x[ix].mean(axis=0) > 0).astype(int) for ix in IDX)

    gp = groups_plus(net)
    ngp = np.stack([groups_plus(nn[s]) for s in range(N_SURR)])
    rows = []
    for j, (f, h, L) in enumerate(GRID):
        rows.append({"семейство": f, "шаг H": h, "окно L": L, "gross, %": gm[j] * 100, "net, %": nm[j] * 100, "сделок/мес": trades[:, j].mean(),
                     "групп net>0": int(gp[j]), "контрактов gross>0": int((gross[:, j] > 0).sum()),
                     "p gross": (ngm[:, j] >= gm[j]).mean(), "нуль gross, %": ngm[:, j].mean() * 100,
                     "gross без NRF6, %": np.delete(gross[:, j], k6).mean() * 100,
                     "безубыточная цена, б.п. за сторону ноги": gross[:, j].mean() / max(turnover[:, j].mean(), 1e-9) / 2 * 1e4})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "spreads_cells.csv", index=False)

    print(f"\nДинамические стратегии на спреде: {len(GRID)} ячеек, суррогатов {N_SURR}")
    for f in FAMILIES:
        sub = df[df["семейство"] == f]
        if f == "fade":
            print("\nfade (позиция против знака последнего бара):\n" + sub[["шаг H", "gross, %", "net, %", "сделок/мес", "групп net>0", "контрактов gross>0", "p gross",
                  "безубыточная цена, б.п. за сторону ноги"]].round(3).to_string(index=False))
            continue
        print(f"\n{f}: gross, % (строки L, столбцы H) / net, %")
        a = sub.pivot(index="окно L", columns="шаг H", values="gross, %").round(1)
        b = sub.pivot(index="окно L", columns="шаг H", values="net, %").round(1)
        print(pd.concat({"gross": a, "net": b}, axis=1).to_string())
    print("\nСводка (реальные vs суррогаты)")
    summ = [
        ("ячеек с net > 0 в >= 4 группах из 5", int((gp >= 4).sum()), (ngp >= 4).sum(axis=1)),
        ("ячеек с net > 0 в 5 группах", int((gp >= 5).sum()), (ngp >= 5).sum(axis=1)),
        ("ячеек со средним net > 0", int((nm > 0).sum()), (nnm > 0).sum(axis=1)),
        ("лучший средний net, %", nm.max() * 100, nnm.max(axis=1) * 100),
        ("лучший средний gross, %", gm.max() * 100, ngm.max(axis=1) * 100),
        ("средний gross по сетке, %", gm.mean() * 100, ngm.mean(axis=1) * 100),
    ]
    for f in FAMILIES:
        m = np.array([g[0] == f for g in GRID])
        summ.append((f"средний gross {f}, %", gm[m].mean() * 100, ngm[:, m].mean(axis=1) * 100))
    srows = []
    for name, real, nul in summ:
        p = (np.asarray(nul) >= real).mean()
        srows.append({"показатель": name, "реальные": round(float(real), 2), "нуль: среднее": round(float(np.mean(nul)), 2),
                      "нуль: 95-й перц.": round(float(np.quantile(nul, .95)), 2), "p-value": round(float(p), 3)})
    sdf = pd.DataFrame(srows)
    sdf.to_csv(REPORTS / "spreads_summary.csv", index=False)
    print(sdf.to_string(index=False))
    j = int(nm.argmax())
    print(f"\nЛучшая по net: {GRID[j]} gross {gm[j]*100:+.2f}%, net {nm[j]*100:+.2f}%, сделок {trades[:, j].mean():.0f}, групп {gp[j]}, "
          f"плюсовых контрактов (net) {(net[:, j] > 0).sum()}/{len(net)}")
    print("Топ-10 по net:")
    print(df.sort_values("net, %", ascending=False).head(10).round(3).to_string(index=False))
    print("\nСтатический carry (длинный ближний/короткий следующий на весь месяц), gross, %:")
    for s in ("зима", "лето", "межсезонье"):
        m = SEASON == s
        print(f"  {s}: {real_carry[m].mean()*100:+.2f}% ({m.sum()} контрактов), суррогаты: среднее {ncar[:, m].mean()*100:+.2f}%, "
              f"p(>=) = {(ncar[:, m].mean(axis=1) >= real_carry[m].mean()).mean():.3f}, p(<=) = {(ncar[:, m].mean(axis=1) <= real_carry[m].mean()).mean():.3f}")
    print(f"  все: {real_carry.mean()*100:+.2f}%, p(>=) = {(ncar.mean(axis=1) >= real_carry.mean()).mean():.3f}, "
          f"p(<=) = {(ncar.mean(axis=1) <= real_carry.mean()).mean():.3f}")
