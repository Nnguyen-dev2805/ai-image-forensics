from __future__ import annotations

from pathlib import Path

from aiforensics.config import load_config


def test_modal_eval_small_config_points_to_volume_paths() -> None:
    cfg = load_config(Path("configs/qwen_ft_protocol_a_small_modal.yaml"))

    assert str(cfg.paths.data_root) == "/vol/data/protocol-a-small"
    assert str(cfg.paths.manifest_root) == "/vol/data/protocol-a-small/manifests"
    assert str(cfg.paths.output_root) == "/vol/eval/protocol-a-small/eval-small"
    assert cfg.baselines.qwen_ft.adapter_uri == "/vol/checkpoints/protocol-a-small/final_adapter"
    assert cfg.datasets.genimage_unseen.manifest == Path(
        "/vol/data/protocol-a-small/manifests/eval_small.csv"
    )
    assert not cfg.baselines.qwen_ft.cache_outputs


def test_modal_eval_full_config_points_to_full_manifest() -> None:
    cfg = load_config(Path("configs/qwen_ft_protocol_a_full_modal.yaml"))

    assert str(cfg.paths.output_root) == "/vol/eval/protocol-a-small/eval-full"
    assert "eval_full" in str(cfg.datasets.genimage_unseen.manifest)
    assert not cfg.baselines.qwen_ft.cache_outputs
