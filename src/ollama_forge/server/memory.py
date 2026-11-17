"""Memory tracker — enforces GPU budget and LRU eviction of subprocess backends."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backends import Backend


class MemoryTracker:
    """
    Budget enforcer and LRU eviction manager.

    Why refcounts: a streaming response holds Metal memory until the last
    byte is delivered. Evicting mid-stream would kill the subprocess the
    client is currently reading from, so any model with in-flight requests
    is excluded from eviction candidates.
    """

    def __init__(self, budget_mb: int) -> None:
        self.budget_mb = budget_mb
        # LRU ordered dict: model_id → estimated_mb
        # why OrderedDict: O(1) move_to_end gives LRU for free
        self._loaded: OrderedDict[str, int] = OrderedDict()
        self._in_flight: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def used_mb(self) -> int:
        return sum(self._loaded.values())

    def _evictable(self) -> list[str]:
        return [mid for mid in self._loaded if not self._in_flight.get(mid)]

    async def admit(self, model_id: str, backends: dict[str, "Backend"], estimated_mb: int) -> None:
        """Ensure model_id's backend is running, evicting LRU backends if needed."""
        async with self._lock:
            if model_id in self._loaded:
                self._loaded.move_to_end(model_id)
                return
            candidates = self._evictable()
            i = 0
            while self.used_mb() + estimated_mb > self.budget_mb and i < len(candidates):
                victim = candidates[i]
                self._loaded.pop(victim)
                b = backends.get(victim)
                if b is not None:
                    await b.stop()
                    del backends[victim]
                i += 1
            b = backends.get(model_id)
            if b is not None:
                await b.start()
            self._loaded[model_id] = estimated_mb

    def mark_in_flight(self, model_id: str) -> None:
        self._in_flight[model_id] = self._in_flight.get(model_id, 0) + 1

    def mark_done(self, model_id: str) -> None:
        count = self._in_flight.get(model_id, 0) - 1
        if count <= 0:
            self._in_flight.pop(model_id, None)
        else:
            self._in_flight[model_id] = count
