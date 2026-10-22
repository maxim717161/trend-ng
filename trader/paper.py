"""Симулятор исполнения лимитных заявок поверх реальных котировок (рыночные данные берутся у настоящего брокера, только чтение).

Заявка исполняется, только если рынок дошёл до её цены: покупка - лучший ask <= цена, продажа - лучший bid >= цена.
Это пессимистичнее реальной очереди (когда сделка по нашей цене могла пройти раньше, чем мы опросили стакан), зато без самообмана.
Состояние (позиция, заявки) хранится в переданном словаре и сохраняется вместе с состоянием движка.
"""
from __future__ import annotations

from datetime import datetime

from trader.broker import OrderNotFound
from trader.models import CANCELLED, FILLED, NEW, Bar, Book, Instrument, OrderState


class PaperBroker:
    def __init__(self, data, st: dict, start_equity: float = 100_000.0):
        self.data, self.st, self.start_equity = data, st, start_equity
        st.setdefault("pos", {})
        st.setdefault("orders", {})
        st.setdefault("seq", 0)

    def list_futures(self) -> list[Instrument]:
        return self.data.list_futures()

    def hourly_bars(self, inst: Instrument, now: datetime, n: int) -> list[Bar]:
        return self.data.hourly_bars(inst, now, n)

    def book(self, inst: Instrument) -> Book:
        return self.data.book(inst)

    def is_trading(self, inst: Instrument) -> bool:
        return self.data.is_trading(inst)

    def position(self, inst: Instrument) -> int:
        return int(self.st["pos"].get(inst.ticker, 0))

    def equity(self) -> float:
        return self.start_equity

    def _try_fill(self, inst: Instrument, o: dict) -> None:
        if o["status"] != NEW:
            return
        b = self.data.book(inst)
        if o["side"] > 0 and b.ask is not None and b.ask <= o["price"]:
            px = min(o["price"], b.ask) if o.get("taker") else o["price"]
        elif o["side"] < 0 and b.bid is not None and b.bid >= o["price"]:
            px = max(o["price"], b.bid) if o.get("taker") else o["price"]
        else:
            return
        o.update(status=FILLED, executed=o["qty"], avg=px)
        self.st["pos"][inst.ticker] = self.position(inst) + o["side"] * o["qty"]

    def _view(self, o: dict) -> OrderState:
        return OrderState(o["id"], o["status"], o["qty"], o["executed"], o["avg"])

    def place_limit(self, inst: Instrument, req_id: str, side: int, qty: int, price: float) -> OrderState:
        for o in self.st["orders"].values():
            if o["req_id"] == req_id:
                return self._view(o)
        self.st["seq"] += 1
        done = [k for k, v in self.st["orders"].items() if v["status"] != NEW]
        for k in done[:-20]:
            del self.st["orders"][k]
        o = {"id": f"paper-{self.st['seq']}", "req_id": req_id, "side": side, "qty": qty, "price": price,
             "status": NEW, "executed": 0, "avg": None, "taker": True}
        self.st["orders"][o["id"]] = o
        self._try_fill(inst, o)
        o["taker"] = False
        return self._view(o)

    def find_active(self, inst: Instrument, req_id: str) -> OrderState | None:
        for o in self.st["orders"].values():
            if o["req_id"] == req_id and o["status"] == NEW:
                return self._view(o)
        return None

    def order_state(self, inst: Instrument, order_id: str) -> OrderState:
        o = self.st["orders"].get(order_id)
        if o is None:
            raise OrderNotFound(order_id)
        self._try_fill(inst, o)
        return self._view(o)

    def cancel(self, inst: Instrument, order_id: str) -> None:
        o = self.st["orders"].get(order_id)
        if o is None:
            raise OrderNotFound(order_id)
        if o["status"] == NEW:
            o["status"] = CANCELLED
