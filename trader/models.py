from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Instrument:
    ticker: str
    uid: str
    expiry: datetime
    tick: float
    lot: int = 1
    figi: str = ""


@dataclass
class Book:
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float] = None

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return self.last
        return (self.bid + self.ask) / 2


@dataclass
class Bar:
    ts: datetime
    open: float
    close: float
    complete: bool = True


NEW, PARTIAL, FILLED, CANCELLED, REJECTED, UNKNOWN = "NEW", "PARTIAL", "FILLED", "CANCELLED", "REJECTED", "UNKNOWN"


@dataclass
class OrderState:
    order_id: str
    status: str
    lots_requested: int = 0
    lots_executed: int = 0
    avg_price: Optional[float] = None


@dataclass
class Event:
    ts: str
    type: str
    data: dict = field(default_factory=dict)
