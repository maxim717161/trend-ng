from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    mode: str = "paper"                 # paper | live
    live_confirm: str = ""
    max_lots: int = 1
    guard_days: int = 3                 # за столько дней до экспирации переходим на следующий контракт
    passive_ttl_min: float = 5.0        # как часто переставляем пассивную лимитку к лучшей цене
    aggressive_after_min: float = 30.0  # через сколько минут после решения идём в «маркетабельную» лимитку
    max_spread_ticks: int = 30          # при более широком спреде пассивную заявку не ставим
    equity_floor_rub: float = 0.0       # ниже этой стоимости счёта закрываем позицию и останавливаемся
    paper_start_equity: float = 100_000.0
    s3_bucket: str = ""
    s3_prefix: str = "ngm-trader/"
    s3_endpoint: str = "https://storage.yandexcloud.net"
    tinvest_token: str = ""
    account_id: str = ""
    tinvest_url: str = "https://invest-public-api.tinkoff.ru/rest/"

    @classmethod
    def from_env(cls) -> "Config":
        e = os.environ
        return cls(
            mode=e.get("MODE", "paper").lower(), live_confirm=e.get("LIVE_CONFIRM", ""),
            max_lots=int(e.get("MAX_LOTS", 1)), guard_days=int(e.get("GUARD_DAYS", 3)),
            passive_ttl_min=float(e.get("PASSIVE_TTL_MIN", 5)), aggressive_after_min=float(e.get("AGGRESSIVE_AFTER_MIN", 30)),
            max_spread_ticks=int(e.get("MAX_SPREAD_TICKS", 30)), equity_floor_rub=float(e.get("EQUITY_FLOOR_RUB", 0)),
            paper_start_equity=float(e.get("PAPER_START_EQUITY", 100000)),
            s3_bucket=e.get("S3_BUCKET", ""), s3_prefix=e.get("S3_PREFIX", "ngm-trader/"),
            s3_endpoint=e.get("S3_ENDPOINT", "https://storage.yandexcloud.net"),
            tinvest_token=e.get("TINVEST_TOKEN", ""), account_id=e.get("TINVEST_ACCOUNT_ID", ""),
            tinvest_url=e.get("TINVEST_URL", "https://invest-public-api.tinkoff.ru/rest/"),
        )

    @property
    def live(self) -> bool:
        return self.mode == "live"

    def validate(self) -> None:
        if self.mode not in ("paper", "live"):
            raise ValueError("MODE должен быть paper или live")
        if self.max_lots != 1:
            raise ValueError("пробный запуск: MAX_LOTS должен быть 1")
        if self.live:
            if self.live_confirm != "I_UNDERSTAND":
                raise ValueError("для MODE=live нужно LIVE_CONFIRM=I_UNDERSTAND")
            if self.equity_floor_rub <= 0:
                raise ValueError("для MODE=live обязателен EQUITY_FLOOR_RUB > 0")
            if not self.account_id:
                raise ValueError("для MODE=live обязателен TINVEST_ACCOUNT_ID")
        if not self.tinvest_token:
            raise ValueError("нужен TINVEST_TOKEN (в paper-режиме достаточно токена только для чтения)")
