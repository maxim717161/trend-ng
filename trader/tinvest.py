"""Адаптер T-Invest API (REST). ВНИМАНИЕ: написан по документации, на реальном счёте не проверялся.
Перед боевым режимом выполнить `python -m trader.run_local selfcheck` (только чтение) и прогнать paper-режим."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import requests

from trader.models import CANCELLED, FILLED, NEW, PARTIAL, REJECTED, UNKNOWN, Bar, Book, Instrument, OrderState
from trader.broker import OrderNotFound

SVC = "tinkoff.public.invest.api.contract.v1."
TICKER_RE = re.compile(r"^NR[FGHJKMNQUVXZ]\d$")
STATUS = {
    "EXECUTION_REPORT_STATUS_FILL": FILLED, "EXECUTION_REPORT_STATUS_NEW": NEW,
    "EXECUTION_REPORT_STATUS_PARTIALLYFILL": PARTIAL, "EXECUTION_REPORT_STATUS_CANCELLED": CANCELLED,
    "EXECUTION_REPORT_STATUS_REJECTED": REJECTED,
}


class BrokerError(Exception):
    pass


def q(v: dict | None) -> float | None:
    if not v:
        return None
    return int(v.get("units", 0)) + int(v.get("nano", 0)) / 1e9


def quotation(x: float) -> dict:
    units = int(x)
    nano = int(round((x - units) * 1e9))
    return {"units": str(units), "nano": nano}


def parse_ts(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "." in s:
        head, tail = s.split(".", 1)
        frac, tz = re.match(r"(\d+)(.*)", tail).groups()
        s = f"{head}.{frac[:6]}{tz}"
    return datetime.fromisoformat(s)


class TInvestBroker:
    def __init__(self, token: str, account_id: str, base_url: str = "https://invest-public-api.tinkoff.ru/rest/", timeout: float = 10.0):
        self.account_id, self.base, self.timeout = account_id, base_url.rstrip("/") + "/", timeout
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})

    def _call(self, service: str, method: str, body: dict) -> dict:
        try:
            r = self.s.post(f"{self.base}{SVC}{service}/{method}", json=body, timeout=self.timeout)
        except requests.RequestException as e:
            raise BrokerError(f"{service}/{method}: {e}") from e
        if r.status_code != 200:
            code = ""
            try:
                code = str(r.json().get("code", ""))
            except ValueError:
                pass
            if code == "50005" or "NOT_FOUND" in r.text.upper():
                raise OrderNotFound(r.text[:300])
            raise BrokerError(f"{service}/{method}: HTTP {r.status_code} {r.text[:300]}")
        return r.json()

    def accounts(self) -> list[dict]:
        return self._call("UsersService", "GetAccounts", {}).get("accounts", [])

    def list_futures(self) -> list[Instrument]:
        r = self._call("InstrumentsService", "Futures", {"instrumentStatus": "INSTRUMENT_STATUS_BASE"})
        out = []
        for f in r.get("instruments", []):
            if not TICKER_RE.match(f.get("ticker", "")) or f.get("classCode") != "SPBFUT":
                continue
            out.append(Instrument(ticker=f["ticker"], uid=f["uid"], figi=f.get("figi", ""), expiry=parse_ts(f["expirationDate"]),
                                  tick=q(f.get("minPriceIncrement")) or 0.001, lot=int(f.get("lot", 1))))
        return sorted(out, key=lambda i: i.expiry)

    def hourly_bars(self, inst: Instrument, now: datetime, n: int) -> list[Bar]:
        bars: dict[datetime, Bar] = {}
        end = now.astimezone(timezone.utc)
        for _ in range(8):
            start = end - timedelta(days=7)
            r = self._call("MarketDataService", "GetCandles", {
                "instrumentId": inst.uid, "interval": "CANDLE_INTERVAL_HOUR",
                "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "to": end.strftime("%Y-%m-%dT%H:%M:%SZ")})
            for c in r.get("candles", []):
                ts = parse_ts(c["time"])
                bars[ts] = Bar(ts, q(c["open"]), q(c["close"]), bool(c.get("isComplete", True)))
            if len(bars) >= n:
                break
            end = start
        return [bars[k] for k in sorted(bars)]

    def book(self, inst: Instrument) -> Book:
        r = self._call("MarketDataService", "GetOrderBook", {"instrumentId": inst.uid, "depth": 1})
        bid = q(r["bids"][0]["price"]) if r.get("bids") else None
        ask = q(r["asks"][0]["price"]) if r.get("asks") else None
        return Book(bid, ask, q(r.get("lastPrice")))

    def is_trading(self, inst: Instrument) -> bool:
        r = self._call("MarketDataService", "GetTradingStatus", {"instrumentId": inst.uid})
        return r.get("tradingStatus") == "SECURITY_TRADING_STATUS_NORMAL_TRADING" and bool(r.get("limitOrderAvailableFlag", True))

    def position(self, inst: Instrument) -> int:
        r = self._call("OperationsService", "GetPositions", {"accountId": self.account_id})
        for f in r.get("futures", []):
            if f.get("instrumentUid") == inst.uid or (inst.figi and f.get("figi") == inst.figi):
                return int(f.get("balance", 0))
        return 0

    def equity(self) -> float:
        r = self._call("OperationsService", "GetPortfolio", {"accountId": self.account_id, "currency": "RUB"})
        return q(r["totalAmountPortfolio"]) or 0.0

    def _state(self, r: dict) -> OrderState:
        price = q(r.get("averagePositionPrice")) or q(r.get("executedOrderPrice"))
        return OrderState(r.get("orderId", ""), STATUS.get(r.get("executionReportStatus"), UNKNOWN),
                          int(r.get("lotsRequested", 0)), int(r.get("lotsExecuted", 0)), price)

    def place_limit(self, inst: Instrument, req_id: str, side: int, qty: int, price: float) -> OrderState:
        """req_id - наш UUID (ключ идемпотентности); в ответе order_id - биржевой номер заявки, по нему дальше работаем."""
        r = self._call("OrdersService", "PostOrder", {
            "instrumentId": inst.uid, "quantity": str(qty), "price": quotation(price),
            "direction": "ORDER_DIRECTION_BUY" if side > 0 else "ORDER_DIRECTION_SELL",
            "accountId": self.account_id, "orderType": "ORDER_TYPE_LIMIT", "orderId": req_id})
        return self._state(r)

    def find_active(self, inst: Instrument, req_id: str) -> OrderState | None:
        r = self._call("OrdersService", "GetOrders", {"accountId": self.account_id})
        for o in r.get("orders", []):
            if o.get("orderRequestId") == req_id:
                return self._state(o)
        return None

    def order_state(self, inst: Instrument, order_id: str) -> OrderState:
        r = self._call("OrdersService", "GetOrderState", {"accountId": self.account_id, "orderId": order_id})
        return self._state(r)

    def cancel(self, inst: Instrument, order_id: str) -> None:
        self._call("OrdersService", "CancelOrder", {"accountId": self.account_id, "orderId": order_id})
