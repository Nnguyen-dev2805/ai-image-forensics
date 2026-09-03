import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def read_qwen_cache(cache_path: Path, sample_id: str, counts: dict) -> str | None:
    if cache_path.exists():
        try:
            raw_text = cache_path.read_text(encoding="utf-8")
            cache_data = json.loads(raw_text)
            if isinstance(cache_data, dict) and isinstance(cache_data.get("raw_output"), str):
                raw_output = cache_data["raw_output"]
                counts["cache_hits"] += 1
                return raw_output
            else:
                counts["cache_misses"] += 1
        except Exception as e:
            logger.warning(f"Cache read failed for {sample_id}: {e}")
            counts["cache_misses"] += 1
    else:
        counts["cache_misses"] += 1
    return None


def write_qwen_cache(cache_path: Path, sample_id: str, raw_output: str):
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(dir=cache_path.parent, prefix=".tmp", text=True)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump({"raw_output": raw_output}, f)
        os.replace(tmp_name, cache_path)
    except Exception as e:
        logger.warning(f"Cache write failed for {sample_id}: {e}")
