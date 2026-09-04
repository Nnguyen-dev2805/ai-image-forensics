"""Shared progress reporting for long-running per-item loops.

Heavy baselines (Qwen-VL, assisted Qwen) process hundreds of images and a full
run can take hours. Without per-item feedback, a notebook cell looks hung and
the only honest answer to "is it working?" is "probably". A progress bar closes
that gap: one line per image would flood cell output and bloat saved notebook
files, so the bar stays a single live line, supports per-item postfix state,
and can also log milestones so a dead session's log file still shows how far it
got.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar

from tqdm import tqdm

__all__ = ["progress_iter"]

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Enough samples for a rate estimate to mean something.
_WARMUP_ITEMS = 5


class _ProgressIterator(Iterator[T], Generic[T]):
    """Iterate ``items`` under a tqdm bar with optional milestone logging."""

    def __init__(
        self,
        bar: tqdm,
        *,
        description: str,
        total: int,
        log_every: int,
    ) -> None:
        self._bar = bar
        self._iter = iter(bar)
        self._description = description
        self._total = total
        self._log_every = log_every
        self._completed = 0
        self._started = time.monotonic()

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        item = next(self._iter)
        self._completed += 1
        if self._log_every > 0 and (
            self._completed % self._log_every == 0 or self._completed == self._total
        ):
            self._log_milestone()
        return item

    def set_postfix(self, **kwargs: str) -> None:
        self._bar.set_postfix(kwargs)

    def _log_milestone(self) -> None:
        elapsed = time.monotonic() - self._started
        completed = self._completed
        rate = elapsed / completed if completed >= _WARMUP_ITEMS else None
        remaining = rate * (self._total - completed) if rate is not None else None
        logger.info(
            "%s: %d/%d done, avg %.1fs/img%s",
            self._description,
            completed,
            self._total,
            elapsed / completed,
            f", ~{remaining / 60:.0f} min remaining" if remaining is not None else "",
        )


def progress_iter(
    description: str,
    items: Sequence[T],
    *,
    unit: str = "img",
    log_every: int = 0,
) -> _ProgressIterator[T]:
    """Return an iterator over ``items`` that drives a live progress bar.

    ``unit`` labels what one step is (an image, a batch). ``log_every``
    additionally logs one line every N items (and at the end) at INFO level, so
    a run's own log file shows milestones even though the bar itself is
    transient terminal state. A non-positive value disables it.
    """
    bar = tqdm(
        items,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
        file=sys.stderr,
    )
    return _ProgressIterator(
        bar,
        description=description,
        total=len(items),
        log_every=log_every,
    )
