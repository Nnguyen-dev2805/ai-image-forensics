"""Tests for shared progress reporting.

The point being pinned: long runs must show liveness, and a dead session's log
file must still record how far the run got. No real terminal output is needed —
tqdm writes to a stub stream.
"""

from __future__ import annotations

import logging

import pytest

from aiforensics.progress import progress_iter


class _CaptureBar:
    """Minimal tqdm stand-in that records postfix updates."""

    def __init__(self, items, **kwargs):
        self._items = list(items)
        self.postfix_updates: list[dict[str, str]] = []
        self.kwargs = kwargs

    def __iter__(self):
        yield from self._items

    def set_postfix(self, ordered_dict=None, **kwargs):
        self.postfix_updates.append(dict(ordered_dict or {}, **kwargs))


@pytest.fixture
def capture_tqdm(monkeypatch: pytest.MonkeyPatch) -> list[_CaptureBar]:
    bars: list[_CaptureBar] = []

    def _factory(items, **kwargs):
        bar = _CaptureBar(items, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr("aiforensics.progress.tqdm", _factory)
    return bars


def test_yields_every_item_in_order(capture_tqdm):
    result = list(progress_iter("test", [1, 2, 3]))
    assert result == [1, 2, 3]


def test_postfix_reaches_the_bar(capture_tqdm):
    """Per-item state (parse counts) must surface on the live bar."""
    iterator = progress_iter("test", ["a", "b"])
    for _item in iterator:
        iterator.set_postfix(parsed=1)
    assert capture_tqdm[0].postfix_updates == [{"parsed": 1}, {"parsed": 1}]


def test_total_is_reported_to_the_bar(capture_tqdm):
    """tqdm derives the total from the sequence itself; the bar gets the items."""
    items = list(range(7))
    list(progress_iter("test", items))
    bar = capture_tqdm[0]
    assert bar._items == items
    assert bar.kwargs["desc"] == "test"
    assert bar.kwargs["unit"] == "img"


def test_milestone_logged_at_interval_and_at_end(capture_tqdm, caplog):
    """A dead session's log file must show how far the run got."""
    records = list(range(10))
    with caplog.at_level(logging.INFO, logger="aiforensics.progress"):
        list(progress_iter("qwen_vl", records, log_every=4))

    messages = [r.message for r in caplog.records]
    # Milestones at 4, 8, and the final 10.
    assert any("4/10" in m for m in messages)
    assert any("8/10" in m for m in messages)
    assert any("10/10" in m for m in messages)


def test_no_milestones_when_log_every_is_zero(capture_tqdm, caplog):
    with caplog.at_level(logging.INFO, logger="aiforensics.progress"):
        list(progress_iter("test", [1, 2, 3], log_every=0))
    assert not caplog.records


def test_milestone_carries_rate_estimate_once_warmed(capture_tqdm, caplog):
    records = list(range(6))
    with caplog.at_level(logging.INFO, logger="aiforensics.progress"):
        list(progress_iter("test", records, log_every=6))
    messages = " ".join(r.message for r in caplog.records)
    assert "avg" in messages


def test_empty_sequence_yields_nothing(capture_tqdm):
    assert list(progress_iter("test", [])) == []
