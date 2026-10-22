from datetime import datetime, timedelta

import pytest

from trader.config import Config
from trader.engine import MSK, Engine, in_session, pick_front
from trader.models import Bar, Book, Instrument
from trader.paper import PaperBroker
from trader.storage import LocalStorage, Store
from trader.strategy import decide

NOW = datetime(2026, 10, 21, 11, 0, tzinfo=MSK)  # среда


class FakeData:
    def __init__(self, now=NOW, trend=1):
        self.insts = [Instrument("NRV6", "uid-v6", now + timedelta(days=20), 0.001, 1, "figi-v6"),
                      Instrument("NRX6", "uid-x6", now + timedelta(days=50), 0.001, 1, "figi-x6")]
        self.bars_for = {}
        self.books = {}
        self.trading = True
        for i in self.insts:
            self.bars_for[i.ticker] = [Bar(now.replace(minute=0) - timedelta(hours=200 - k), 3 + trend * 0.01 * k, 3 + trend * 0.01 * k) for k in range(200)]
            self.books[i.ticker] = Book(5.000, 5.010, 5.005)

    def list_futures(self):
        return self.insts

    def hourly_bars(self, inst, now, n):
        return self.bars_for[inst.ticker]

    def book(self, inst):
        return self.books[inst.ticker]

    def is_trading(self, inst):
        return self.trading


def make(tmp_path, **cfg_kw):
    cfg = Config(tinvest_token="x", **cfg_kw)
    data = FakeData()
    store = Store(LocalStorage(str(tmp_path)), "t/")
    eng = Engine(cfg, PaperBroker(data, {}), store)
    return eng, data, store


def types(ev):
    return [e["type"] for e in ev]


def test_strategy_votes():
    up = [float(i) for i in range(200)]
    assert decide(up, 0)[0] == 1
    assert decide(up[::-1], 0)[0] == -1
    assert decide(up[:50], -1)[0] == -1
    mixed = [10.0] * 200
    for k, v in ((72, 9.0), (96, 11.0), (120, 9.0), (166, 11.0)):
        mixed[-1 - k] = v
    t, info = decide(mixed, -1)
    assert info["votes"] == [1, -1, 1, -1] and t == -1
    three = list(mixed)
    three[-1 - 96] = 9.0
    assert decide(three, -1)[0] == 1


def test_pick_front_and_session():
    d = FakeData()
    assert pick_front(d.insts, NOW, 3).ticker == "NRV6"
    assert pick_front(d.insts, NOW + timedelta(days=18), 3).ticker == "NRX6"
    assert in_session(NOW) and not in_session(NOW.replace(hour=14, minute=2)) and not in_session(NOW + timedelta(days=3))


def test_decision_and_passive_order_then_fill(tmp_path):
    eng, data, store = make(tmp_path)
    ev = eng.tick(NOW)
    assert "decision" in types(ev) and "order_placed" in types(ev)
    placed = [e for e in ev if e["type"] == "order_placed"][0]
    assert placed["side"] == 1 and placed["price"] == 5.0 and placed["phase"] == "passive"
    # рынок не дошёл до цены: заявка висит, повторно не ставится
    ev = eng.tick(NOW + timedelta(minutes=1))
    assert "order_placed" not in types(ev) and "fill" not in types(ev)
    # ask опустился до нашей цены: исполнение
    data.books["NRV6"] = Book(4.99, 5.000, 5.0)
    ev = eng.tick(NOW + timedelta(minutes=2))
    fill = [e for e in ev if e["type"] == "fill"][0]
    assert fill["lots"] == 1 and fill["price"] == 5.0
    st = store.load_state()
    assert st["expected_pos"] == 1 and st["order"] is None and eng.broker.position(data.insts[0]) == 1
    ev = eng.tick(NOW + timedelta(minutes=3))
    assert types(ev) == []


def test_reprice_after_ttl_and_aggressive_phase(tmp_path):
    eng, data, store = make(tmp_path, passive_ttl_min=5, aggressive_after_min=30)
    eng.tick(NOW)
    data.books["NRV6"] = Book(5.02, 5.03, 5.025)
    ev = eng.tick(NOW + timedelta(minutes=2))
    assert "order_placed" not in types(ev)  # ttl ещё не прошёл
    ev = eng.tick(NOW + timedelta(minutes=6))
    placed = [e for e in ev if e["type"] == "order_placed"]
    assert "order_cancel" in types(ev) and placed and placed[0]["price"] == 5.02 and placed[0]["phase"] == "passive"
    data.books["NRV6"] = Book(5.03, 5.04, 5.035)
    ev = eng.tick(NOW + timedelta(minutes=31))
    placed = [e for e in ev if e["type"] == "order_placed"]
    assert placed and placed[0]["phase"] == "aggressive" and placed[0]["price"] == 5.04
    assert "fill" in types(ev)  # маркетабельная лимитка исполнилась по ask
    fill = [e for e in ev if e["type"] == "fill"][0]
    assert fill["slippage_bps"] is not None


def test_wide_spread_blocks_passive_order(tmp_path):
    eng, data, _ = make(tmp_path, max_spread_ticks=30)
    data.books["NRV6"] = Book(5.0, 5.2, 5.1)
    ev = eng.tick(NOW)
    assert "wide_spread" in types(ev) and "order_placed" not in types(ev)


def test_outside_session_does_nothing(tmp_path):
    eng, data, _ = make(tmp_path)
    ev = eng.tick(NOW.replace(hour=3))
    assert "heartbeat" in types(ev) and "order_placed" not in types(ev) and "decision" not in types(ev)


def test_kill_switch_and_flatten(tmp_path):
    eng, data, store = make(tmp_path)
    store.b.put("t/control/halt", b"1")
    ev = eng.tick(NOW)
    assert "order_placed" not in types(ev)


def test_flatten_control_closes_position(tmp_path):
    eng, data, store = make(tmp_path)
    eng.tick(NOW)
    data.books["NRV6"] = Book(4.99, 5.0, 5.0)
    eng.tick(NOW + timedelta(minutes=1))
    assert store.load_state()["expected_pos"] == 1
    store.b.put("t/control/flatten", b"1")
    data.books["NRV6"] = Book(5.0, 5.01, 5.0)
    ev = eng.tick(NOW + timedelta(minutes=2))
    placed = [e for e in ev if e["type"] == "order_placed"][0]
    assert placed["side"] == -1 and placed["price"] == 5.01
    data.books["NRV6"] = Book(5.02, 5.03, 5.0)
    eng.tick(NOW + timedelta(minutes=3))
    assert store.load_state()["expected_pos"] == 0


def test_position_mismatch_halts_after_two_ticks(tmp_path):
    eng, data, store = make(tmp_path)
    eng.tick(NOW)
    eng.broker.st["pos"]["NRV6"] = 1  # позиция появилась мимо нас (ручная сделка)
    store.save_state({**store.load_state(), "paper": eng.broker.st, "order": None})
    ev1 = eng.tick(NOW + timedelta(minutes=1))
    assert "position_mismatch" in types(ev1)
    ev2 = eng.tick(NOW + timedelta(minutes=2))
    assert "safety_halt" in types(ev2)
    ev3 = eng.tick(NOW + timedelta(minutes=3))
    assert "order_placed" not in types(ev3)


def test_roll_with_position_flattens_then_switches(tmp_path):
    eng, data, store = make(tmp_path)
    eng.tick(NOW)
    data.books["NRV6"] = Book(4.99, 5.0, 5.0)
    eng.tick(NOW + timedelta(minutes=1))
    assert store.load_state()["expected_pos"] == 1
    later = NOW + timedelta(days=19)  # понедельник, до экспирации NRV6 остался 1 день -> рабочий контракт NRX6
    data.books["NRV6"] = Book(5.1, 5.11, 5.1)
    ev = eng.tick(later)
    placed = [e for e in ev if e["type"] == "order_placed"]
    assert placed and placed[0]["side"] == -1 and placed[0]["contract"] == "NRV6"
    data.books["NRV6"] = Book(5.12, 5.13, 5.1)
    ev = eng.tick(later + timedelta(minutes=1))
    assert "roll" in types(ev) and store.load_state()["contract"] == "NRX6"
    assert [e for e in ev if e["type"] == "fill"][0]["contract"] == "NRV6"


def test_config_validation():
    with pytest.raises(ValueError):
        Config(mode="live", tinvest_token="x", account_id="a", equity_floor_rub=1000).validate()
    Config(mode="live", live_confirm="I_UNDERSTAND", tinvest_token="x", account_id="a", equity_floor_rub=1000).validate()
    with pytest.raises(ValueError):
        Config(mode="live", live_confirm="I_UNDERSTAND", tinvest_token="x", account_id="a").validate()
    with pytest.raises(ValueError):
        Config(tinvest_token="x", max_lots=2).validate()
