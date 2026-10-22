"""Пробная стратегия: многодневный моментум-ансамбль на часовых барах.

Четыре голоса: знак изменения цены закрытия за K баров (K = 72, 96, 120, 166 часовых баров, это окна 4320-10000 минут
из исследования моментума). Позиция +1/-1 при согласии >= 3 из 4 голосов, иначе сохраняется предыдущая позиция
(гистерезис, чтобы не крутить комиссию на спорных сигналах). Чистая функция без зависимостей: используется и в бэктесте, и в боевом коде.
"""
from __future__ import annotations

from typing import Sequence

KS: tuple[int, ...] = (72, 96, 120, 166)
ENTER = 0.5


def decide(closes: Sequence[float], prev_target: int, ks: Sequence[int] = KS, enter: float = ENTER) -> tuple[int, dict]:
    """closes - закрытия завершённых часовых баров по возрастанию времени. Возвращает (целевая позиция, диагностика)."""
    n = len(closes)
    votes = []
    for k in ks:
        if n > k:
            d = closes[-1] - closes[-1 - k]
            votes.append(1 if d > 0 else -1 if d < 0 else 0)
        else:
            votes.append(0)
    score = sum(votes) / len(ks)
    if score >= enter:
        target = 1
    elif score <= -enter:
        target = -1
    else:
        target = prev_target
    return target, {"votes": votes, "score": score, "bars": n, "last_close": closes[-1] if n else None}


def required_bars(ks: Sequence[int] = KS) -> int:
    return max(ks) + 1
