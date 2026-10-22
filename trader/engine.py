"""Один «тик» торгового цикла (вызывается таймером раз в минуту).

Порядок: состояние -> режим (halt/flatten) -> контракт (ролловер) -> торговая сессия -> синхронизация заявки и позиции ->
решение на новом закрытом часовом баре -> выставление/перестановка лимитной заявки -> сохранение состояния и событий.
Всё состояние лежит в state.json; позиция брокера - источник истины, расхождение с учётом по нашим исполнениям останавливает торговлю.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from trader.broker import Broker, OrderNotFound
from trader.config import Config
from trader.models import CANCELLED, FILLED, NEW, PARTIAL, REJECTED, UNKNOWN, Instrument, OrderState
from trader.storage import Store
from trader.strategy import decide, required_bars
from trader.tinvest import BrokerError

MSK = ZoneInfo("Europe/Moscow")
SESSIONS = [((9, 5), (13, 55)), ((14, 10), (18, 40)), ((19, 10), (23, 45))]
DONE = (FILLED, CANCELLED, REJECTED)


def in_session(now: datetime) -> bool:
    n = now.astimezone(MSK)
    if n.weekday() >= 5:
        return False
    hm = (n.hour, n.minute)
    return any(a <= hm <= b for a, b in SESSIONS)


def round_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 10)


def inst_to_dict(i: Instrument) -> dict:
    return {"ticker": i.ticker, "uid": i.uid, "figi": i.figi, "expiry": i.expiry.isoformat(), "tick": i.tick, "lot": i.lot}


def inst_from_dict(d: dict) -> Instrument:
    return Instrument(d["ticker"], d["uid"], datetime.fromisoformat(d["expiry"]), d["tick"], d.get("lot", 1), d.get("figi", ""))


def pick_front(insts: list[Instrument], now: datetime, guard_days: int) -> Instrument | None:
    today = now.astimezone(MSK).date()
    for i in sorted(insts, key=lambda x: x.expiry):
        if (i.expiry.astimezone(MSK).date() - today).days > guard_days:
            return i
    return None


class Engine:
    def __init__(self, cfg: Config, broker: Broker, store: Store):
        self.cfg, self.broker, self.store = cfg, broker, store
        self.ev: list[dict] = []
        self.now = datetime.now(MSK)

    def log(self, type_: str, **data) -> None:
        self.ev.append({"ts": self.now.astimezone(MSK).isoformat(timespec="seconds"), "type": type_, "mode": self.cfg.mode, **data})

    def tick(self, now: datetime | None = None) -> list[dict]:
        self.now = (now or datetime.now(MSK)).astimezone(MSK)
        self.ev = []
        st = self.store.load_state()
        for k, v in (("contract", None), ("target", 0), ("order", None), ("expected_pos", 0), ("last_bar_ts", None),
                     ("checked_hour", None), ("decision", None), ("instruments", []), ("inst_day", None), ("paper", {}),
                     ("safety_halt", None), ("flatten_reason", None), ("hb_hour", None), ("wide_hour", None)):
            st.setdefault(k, v)
        if hasattr(self.broker, "st"):
            self.broker.st = st["paper"]
            self.broker.st.setdefault("pos", {}); self.broker.st.setdefault("orders", {}); self.broker.st.setdefault("seq", 0)
        try:
            self._run(st)
        except (BrokerError, OrderNotFound) as e:
            self.log("error", error=str(e)[:500])
        except Exception as e:  # noqa: BLE001 - тик не должен падать молча: пишем в лог и сохраняем состояние
            self.log("exception", error=repr(e)[:500])
        finally:
            st["last_tick"] = self.now.isoformat(timespec="seconds")
            self.store.save_state(st)
            self.store.write_events(self.now.isoformat(timespec="seconds"), self.ev)
        return self.ev

    def _run(self, st: dict) -> None:
        now = self.now
        ctl = self.store.control()
        halted = ctl == "halt" or st["safety_halt"]
        flatten = ctl == "flatten" or bool(st["flatten_reason"])
        hour = now.strftime("%Y-%m-%dT%H")
        if st["hb_hour"] != hour:
            st["hb_hour"] = hour
            self.log("heartbeat", contract=st["contract"], target=st["target"], expected_pos=st["expected_pos"],
                     halted=bool(halted), flatten=flatten, control=ctl, safety_halt=st["safety_halt"], flatten_reason=st["flatten_reason"])

        if st["inst_day"] != now.date().isoformat() or not st["instruments"]:
            st["instruments"] = [inst_to_dict(i) for i in self.broker.list_futures()]
            st["inst_day"] = now.date().isoformat()
        insts = [inst_from_dict(d) for d in st["instruments"]]
        by_ticker = {i.ticker: i for i in insts}
        front = pick_front(insts, now, self.cfg.guard_days)
        if front is None:
            self.log("no_contract")
            return
        if st["contract"] is None:
            st["contract"] = front.ticker
            self.log("contract_selected", contract=front.ticker)
        inst = by_ticker.get(st["contract"])
        if inst is None:
            if st["expected_pos"] != 0:
                st["safety_halt"] = f"контракт {st['contract']} исчез из списка при позиции {st['expected_pos']}"
                self.log("safety_halt", reason=st["safety_halt"])
                return
            st["contract"], st["order"], st["last_bar_ts"], st["target"] = front.ticker, None, None, 0
            inst = front

        if halted:
            if st["order"] and st["order"].get("broker_id") and in_session(now):
                self._cancel(inst, st)
            return
        if not in_session(now) or not self.broker.is_trading(inst):
            return

        self._sync_order(inst, st)
        pos = self.broker.position(inst)
        if abs(pos) > self.cfg.max_lots:
            st["safety_halt"] = f"позиция {pos} превышает лимит {self.cfg.max_lots}"
            self.log("safety_halt", reason=st["safety_halt"], broker_pos=pos)
            return
        if st["order"] is None and pos != st["expected_pos"]:
            st["mismatch_n"] = st.get("mismatch_n", 0) + 1
            self.log("position_mismatch", broker_pos=pos, expected_pos=st["expected_pos"], n=st["mismatch_n"])
            if st["mismatch_n"] >= 2:
                st["safety_halt"] = f"позиция брокера {pos} не совпадает с учётом по исполнениям {st['expected_pos']}"
                self.log("safety_halt", reason=st["safety_halt"], broker_pos=pos, expected_pos=st["expected_pos"])
            return
        st["mismatch_n"] = 0

        if self.cfg.live and not st["flatten_reason"] and self.cfg.equity_floor_rub > 0:
            eq = self.broker.equity()
            if eq < self.cfg.equity_floor_rub:
                st["flatten_reason"] = f"стоимость счёта {eq:.0f} ниже порога {self.cfg.equity_floor_rub:.0f}"
                flatten = True
                self.log("equity_floor", equity=eq, floor=self.cfg.equity_floor_rub)

        force_flat = False
        if front.ticker != inst.ticker:
            if st["expected_pos"] == 0 and st["order"] is None:
                self.log("roll", old=inst.ticker, new=front.ticker)
                st.update(contract=front.ticker, last_bar_ts=None, target=0, checked_hour=None, decision=None)
                inst = front
            else:
                force_flat = True

        if not flatten and not force_flat:
            self._decide(inst, st)
        eff_target = 0 if (flatten or force_flat) else st["target"]
        if (flatten or force_flat) and st["decision"] is None:
            book = self.broker.book(inst)
            st["decision"] = {"ts": now.isoformat(timespec="seconds"), "mid": book.mid, "bar_ts": None, "reason": "flatten" if flatten else "roll"}
        self._execute(inst, st, eff_target)

    def _decide(self, inst: Instrument, st: dict) -> None:
        now = self.now
        hour = now.strftime("%Y-%m-%dT%H")
        if st["checked_hour"] == hour:
            return
        bars = self.broker.hourly_bars(inst, now, required_bars() + 5)
        done = [b for b in bars if b.complete and b.ts + timedelta(hours=1) <= now]
        st["checked_hour"] = hour
        if not done:
            self.log("no_bars", contract=inst.ticker)
            return
        newest = done[-1].ts.isoformat()
        if newest == st["last_bar_ts"]:
            return
        target, info = decide([b.close for b in done], st["target"])
        book = self.broker.book(inst)
        st["last_bar_ts"] = newest
        self.log("decision", contract=inst.ticker, bar_ts=newest, prev_target=st["target"], target=target, bid=book.bid, ask=book.ask, **info)
        if target != st["target"]:
            st["target"] = target
            st["decision"] = {"ts": now.isoformat(timespec="seconds"), "mid": book.mid, "bar_ts": newest, "reason": "signal"}

    def _apply(self, inst: Instrument, st: dict, s: OrderState) -> None:
        o = st["order"]
        delta = s.lots_executed - o["executed"]
        if delta > 0:
            avg = s.avg_price if s.avg_price is not None else o["price"]
            inc = (avg * s.lots_executed - o["avg"] * o["executed"]) / delta if o["executed"] else avg
            st["expected_pos"] += o["side"] * delta
            dec = st.get("decision") or {}
            dmid = dec.get("mid")
            slip = o["side"] * (inc - dmid) / dmid * 1e4 if dmid else None
            self.log("fill", contract=inst.ticker, side=o["side"], lots=delta, price=inc, order_price=o["price"], phase=o["phase"],
                     decision_mid=dmid, slippage_bps=slip, expected_pos=st["expected_pos"], decision_ts=dec.get("ts"),
                     seconds_since_decision=(self.now - datetime.fromisoformat(dec["ts"])).total_seconds() if dec.get("ts") else None,
                     reason=dec.get("reason"))
            o["executed"], o["avg"] = s.lots_executed, avg
        if s.status in DONE:
            self.log("order_done", status=s.status, executed=o["executed"], requested=o["qty"], price=o["price"], phase=o["phase"])
            st["order"] = None

    def _sync_order(self, inst: Instrument, st: dict) -> None:
        o = st["order"]
        if not o:
            return
        if not o.get("broker_id"):
            s = self.broker.find_active(inst, o["req_id"])
            if s is None:
                self.log("order_lost_unplaced", req_id=o["req_id"])
                st["order"] = None
                return
            o["broker_id"] = s.order_id
        try:
            s = self.broker.order_state(inst, o["broker_id"])
        except OrderNotFound:
            self.log("order_not_found", broker_id=o["broker_id"])
            st["order"] = None
            return
        if s.status == UNKNOWN:
            self.log("order_unknown_status", broker_id=o["broker_id"])
            return
        self._apply(inst, st, s)

    def _cancel(self, inst: Instrument, st: dict) -> None:
        o = st["order"]
        try:
            self.broker.cancel(inst, o["broker_id"])
        except OrderNotFound:
            pass
        self.log("order_cancel", broker_id=o["broker_id"], price=o["price"], phase=o["phase"])
        try:
            self._apply(inst, st, self.broker.order_state(inst, o["broker_id"]))
        except OrderNotFound:
            st["order"] = None

    def _execute(self, inst: Instrument, st: dict, eff_target: int) -> None:
        now = self.now
        need = eff_target - st["expected_pos"]
        o = st["order"]
        if o is not None:
            remaining = o["qty"] - o["executed"]
            if need == 0 or (need > 0) != (o["side"] > 0) or abs(need) != remaining:
                self._cancel(inst, st)
                if st["order"] is not None:
                    return
                need = eff_target - st["expected_pos"]
        if need == 0:
            if st["order"] is None:
                st["decision"] = None
            return
        dec = st["decision"] or {"ts": now.isoformat(timespec="seconds")}
        age_dec = (now - datetime.fromisoformat(dec["ts"])).total_seconds() / 60
        aggressive = age_dec >= self.cfg.aggressive_after_min
        book = self.broker.book(inst)
        if book.bid is None or book.ask is None:
            self.log("no_book", contract=inst.ticker)
            return
        side = 1 if need > 0 else -1
        spread_ticks = (book.ask - book.bid) / inst.tick
        if aggressive:
            price = book.ask if side > 0 else book.bid
        else:
            price = book.bid if side > 0 else book.ask
            if spread_ticks > self.cfg.max_spread_ticks:
                hour = now.strftime("%Y-%m-%dT%H")
                if st["wide_hour"] != hour:
                    st["wide_hour"] = hour
                    self.log("wide_spread", spread_ticks=spread_ticks, bid=book.bid, ask=book.ask)
                return
        price = round_tick(price, inst.tick)
        if st["order"] is not None:
            o = st["order"]
            ttl = 1.0 if aggressive else self.cfg.passive_ttl_min
            placed_age = (now - datetime.fromisoformat(o["placed_at"])).total_seconds() / 60
            if price == o["price"] or placed_age < ttl:
                return
            self._cancel(inst, st)
            need = eff_target - st["expected_pos"]
            if need == 0 or st["order"] is not None:
                return
            side = 1 if need > 0 else -1
        qty = abs(need)
        st["order"] = {"req_id": str(uuid.uuid4()), "broker_id": None, "side": side, "qty": qty, "price": price, "executed": 0, "avg": 0.0,
                       "placed_at": now.isoformat(timespec="seconds"), "phase": "aggressive" if aggressive else "passive"}
        self.store.save_state(st)
        s = self.broker.place_limit(inst, st["order"]["req_id"], side, qty, price)
        st["order"]["broker_id"] = s.order_id
        self.log("order_placed", contract=inst.ticker, side=side, lots=qty, price=price, bid=book.bid, ask=book.ask,
                 phase=st["order"]["phase"], status=s.status, broker_id=s.order_id, minutes_since_decision=round(age_dec, 1))
        if s.status != NEW:
            self._apply(inst, st, s)
