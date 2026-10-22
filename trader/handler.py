"""Точка входа для Yandex Cloud Functions: handler = trader.handler.handler; вызывается триггером-таймером раз в минуту."""
from __future__ import annotations

import json

from trader.config import Config
from trader.engine import Engine
from trader.paper import PaperBroker
from trader.storage import S3Storage, Store
from trader.tinvest import TInvestBroker


def build_engine(cfg: Config | None = None, backend=None) -> Engine:
    cfg = cfg or Config.from_env()
    cfg.validate()
    if backend is None:
        backend = S3Storage(cfg.s3_bucket, cfg.s3_endpoint)
    tin = TInvestBroker(cfg.tinvest_token, cfg.account_id, cfg.tinvest_url)
    broker = tin if cfg.live else PaperBroker(tin, {}, cfg.paper_start_equity)
    return Engine(cfg, broker, Store(backend, cfg.s3_prefix))


def handler(event=None, context=None):
    events = build_engine().tick()
    return {"statusCode": 200, "body": json.dumps({"events": len(events), "types": [e["type"] for e in events]})}
