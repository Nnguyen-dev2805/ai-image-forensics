import hashlib
import json
import re

import pytest

from aiforensics.cache.keys import cache_key


def test_returns_lowercase_64_char_hex_digest():
    key = cache_key({"model": "clip", "checksum": "abc"})
    assert re.fullmatch(r"[0-9a-f]{64}", key) is not None


def test_same_mapping_same_key_across_repeated_calls():
    parts = {"model": "clip", "checksum": "abc"}
    assert cache_key(parts) == cache_key(parts)
    assert cache_key(dict(parts)) == cache_key(dict(parts))


def test_insertion_order_does_not_affect_key():
    assert cache_key({"model": "clip", "checksum": "abc"}) == cache_key(
        {"checksum": "abc", "model": "clip"}
    )


def test_changing_one_value_changes_key():
    assert cache_key({"model": "clip", "checksum": "abc"}) != cache_key(
        {"model": "clip", "checksum": "abd"}
    )


def test_changing_one_key_name_changes_key():
    assert cache_key({"model": "clip", "checksum": "abc"}) != cache_key(
        {"model_id": "clip", "checksum": "abc"}
    )


def test_separator_values_do_not_create_ambiguous_collisions():
    first = cache_key({"a": "b:c", "b": "c"})
    second = cache_key({"a": "b", "b": "c:c"})
    assert first != second


def test_unicode_values_are_deterministic():
    parts = {"model": "Qwen2.5-VL-日本語", "prompt_id": "qwen_json_v1"}
    expected = hashlib.sha256(
        json.dumps(
            dict(sorted(parts.items())),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert cache_key(parts) == expected
    assert cache_key(parts) == cache_key(dict(reversed(list(parts.items()))))


def test_empty_mapping_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        cache_key({})


def test_non_string_keys_are_rejected():
    with pytest.raises(TypeError, match="keys must be strings"):
        cache_key({1: "a"})


def test_non_string_values_are_rejected():
    with pytest.raises(TypeError, match="values must be strings"):
        cache_key({"model": 123})
