"""Локальный запуск: python -m trader.run_local selfcheck | once | loop [секунды]

Хранилище: S3 (если задан S3_BUCKET) или локальная папка LOCAL_DIR (по умолчанию ./trader_local).
selfcheck - только чтение: счета, список контрактов, стакан, свечи, позиция. Запустить до первого боевого включения.
"""
from __future__ import annotations

import os
import sys
import time

from trader.config import Config
from trader.engine import MSK, in_session, pick_front
from trader.handler import build_engine
from trader.storage import LocalStorage, S3Storage
from trader.strategy import decide, required_bars
from trader.tinvest import TInvestBroker
from datetime import datetime


def backend_for(cfg: Config):
    if cfg.s3_bucket:
        return S3Storage(cfg.s3_bucket, cfg.s3_endpoint)
    return LocalStorage(os.environ.get("LOCAL_DIR", "trader_local"))


def selfcheck(cfg: Config) -> None:
    b = TInvestBroker(cfg.tinvest_token, cfg.account_id, cfg.tinvest_url)
    now = datetime.now(MSK)
    print("сейчас (МСК):", now.isoformat(timespec="seconds"), "| торговая сессия по расписанию:", in_session(now))
    for a in b.accounts():
        print("счёт:", a.get("id"), a.get("type"), a.get("name"), a.get("status"))
    futs = b.list_futures()
    print("контрактов NR*:", [f"{f.ticker} {f.expiry.date()} tick={f.tick}" for f in futs])
    front = pick_front(futs, now, cfg.guard_days)
    print("рабочий контракт:", front.ticker if front else None)
    if not front:
        return
    print("торгуется:", b.is_trading(front), "| стакан:", b.book(front))
    bars = b.hourly_bars(front, now, required_bars() + 5)
    done = [x for x in bars if x.complete]
    print(f"часовых баров: {len(bars)} (завершённых {len(done)}, нужно {required_bars()}), последние:", [(x.ts.isoformat(), x.close) for x in bars[-3:]])
    if done:
        print("сигнал сейчас:", decide([x.close for x in done], 0))
    if cfg.account_id:
        print("позиция по контракту (знак: плюс - лонг, минус - шорт; проверьте на открытой позиции!):", b.position(front))
        print("стоимость счёта:", b.equity())


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "once"
    cfg = Config.from_env()
    if cmd == "selfcheck":
        if not cfg.tinvest_token:
            raise SystemExit("нужен TINVEST_TOKEN")
        selfcheck(cfg)
        return
    eng = build_engine(cfg, backend_for(cfg))
    if cmd == "once":
        for e in eng.tick():
            print(e)
    elif cmd == "loop":
        period = float(sys.argv[2]) if len(sys.argv) > 2 else 60
        while True:
            for e in eng.tick():
                print(e, flush=True)
            time.sleep(period)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
