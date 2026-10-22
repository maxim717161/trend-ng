from datetime import datetime, timezone

from trader.models import FILLED, NEW, Instrument
from trader.report import round_trips
from trader.tinvest import TInvestBroker, parse_ts, q, quotation

import pandas as pd


def test_quotation_roundtrip():
    for x in (3.456, 0.001, 12.0, 100.125):
        assert abs(q(quotation(x)) - x) < 1e-9
    assert quotation(3.5) == {"units": "3", "nano": 500000000}


def test_parse_ts():
    assert parse_ts("2026-10-21T10:00:00Z") == datetime(2026, 10, 21, 10, tzinfo=timezone.utc)
    assert parse_ts("2026-10-28T00:00:00.123456789Z").microsecond == 123456


class Fake(TInvestBroker):
    def __init__(self, responses):
        super().__init__("t", "acc")
        self.responses, self.calls = responses, []

    def _call(self, service, method, body):
        self.calls.append((service, method, body))
        return self.responses[method]


INST = Instrument("NRV6", "uid", datetime(2026, 10, 28, tzinfo=timezone.utc), 0.001, 1, "figi")


def test_futures_filter_and_sort():
    r = {"Futures": {"instruments": [
        {"ticker": "NRX6", "classCode": "SPBFUT", "uid": "b", "expirationDate": "2026-11-25T00:00:00Z", "minPriceIncrement": {"units": "0", "nano": 1000000}},
        {"ticker": "SiZ6", "classCode": "SPBFUT", "uid": "x", "expirationDate": "2026-12-15T00:00:00Z"},
        {"ticker": "NRV6", "classCode": "SPBFUT", "uid": "a", "expirationDate": "2026-10-28T00:00:00Z", "minPriceIncrement": {"units": "0", "nano": 1000000}}]}}
    out = Fake(r).list_futures()
    assert [i.ticker for i in out] == ["NRV6", "NRX6"] and out[0].tick == 0.001


def test_book_position_orders():
    r = {"GetOrderBook": {"bids": [{"price": {"units": "3", "nano": 500000000}, "quantity": "5"}], "asks": [{"price": {"units": "3", "nano": 510000000}}], "lastPrice": {"units": "3", "nano": 505000000}},
         "GetPositions": {"futures": [{"instrumentUid": "uid", "balance": "-1"}]},
         "PostOrder": {"orderId": "ex-1", "executionReportStatus": "EXECUTION_REPORT_STATUS_NEW", "lotsRequested": "1", "lotsExecuted": "0"},
         "GetOrderState": {"orderId": "ex-1", "executionReportStatus": "EXECUTION_REPORT_STATUS_FILL", "lotsRequested": "1", "lotsExecuted": "1",
                           "averagePositionPrice": {"currency": "rub", "units": "3", "nano": 500000000}},
         "GetOrders": {"orders": [{"orderId": "ex-2", "orderRequestId": "req-9", "executionReportStatus": "EXECUTION_REPORT_STATUS_NEW", "lotsRequested": "1"}]}}
    b = Fake(r)
    book = b.book(INST)
    assert (book.bid, book.ask) == (3.5, 3.51)
    assert b.position(INST) == -1
    s = b.place_limit(INST, "req-1", -1, 1, 3.51)
    assert s.status == NEW and s.order_id == "ex-1"
    body = b.calls[-1][2]
    assert body["direction"] == "ORDER_DIRECTION_SELL" and body["orderId"] == "req-1" and body["orderType"] == "ORDER_TYPE_LIMIT" and body["price"]["nano"] == 510000000
    st = b.order_state(INST, "ex-1")
    assert st.status == FILLED and st.avg_price == 3.5 and st.lots_executed == 1
    assert b.find_active(INST, "req-9").order_id == "ex-2" and b.find_active(INST, "none") is None


def test_round_trips_with_flip():
    ts = pd.date_range("2026-10-21", periods=3, freq="h")
    fills = pd.DataFrame({"ts": ts, "side": [1, -1, 1], "lots": [1, 2, 1], "price": [100.0, 110.0, 100.0]})
    rt = round_trips(fills)
    assert len(rt) == 2
    assert round(rt["gross, %"].iloc[0], 3) == 10.0
    assert round(rt["gross, %"].iloc[1], 3) == round(10 / 110 * 100, 3)
