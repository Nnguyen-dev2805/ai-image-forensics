from pathlib import Path

import pytest
import yaml

from aiforensics.config.load import ConfigError, load_config
from aiforensics.config.models import AppConfig

SMOKE_CONFIG = "configs/phase_ab_smoke.yaml"
FULL_CONFIG = "configs/phase_ab.yaml"


def test_load_smoke_config_returns_appconfig():
    config = load_config(SMOKE_CONFIG)
    assert isinstance(config, AppConfig)


def test_smoke_config_paths_resolved():
    config = load_config(SMOKE_CONFIG)
    assert config.paths.data_root.is_absolute()
    assert config.paths.data_root.name == "smoke_data"


def test_smoke_config_project_values():
    config = load_config(SMOKE_CONFIG)
    assert config.project.name == "ai-image-forensics"
    assert config.project.phase == "phase_ab_smoke"


def test_smoke_config_baseline_values():
    config = load_config(SMOKE_CONFIG)
    assert config.baselines.clip_probe.enabled is True
    assert config.baselines.clip_probe.model_family == "synthetic"
    assert config.baselines.qwen_vl.enabled is False
    assert config.baselines.npr.allow_deferred is True


def test_full_config_loads_optional_npr_fields():
    config = load_config(FULL_CONFIG)
    assert config.baselines.npr.repo_commit is None
    assert config.baselines.npr.checkpoint_sha256 is None


def test_missing_top_level_section(tmp_path):
    # Make a config missing "paths"
    bad_config = tmp_path / "bad.yaml"
    (tmp_path / "pyproject.toml").touch()
    with open(SMOKE_CONFIG, "r") as f:
        data = yaml.safe_load(f)
    if "paths" in data:
        del data["paths"]

    with open(bad_config, "w") as f:
        yaml.safe_dump(data, f)

    with pytest.raises(ConfigError, match="paths"):
        load_config(bad_config)


def test_invalid_numeric_value(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    (tmp_path / "pyproject.toml").touch()
    with open(SMOKE_CONFIG, "r") as f:
        data = yaml.safe_load(f)

    data["runtime"]["batch_size"] = 0

    with open(bad_config, "w") as f:
        yaml.safe_dump(data, f)

    with pytest.raises(ConfigError) as exc_info:
        load_config(bad_config)
    assert "batch_size" in str(exc_info.value)


def test_missing_config_file():
    with pytest.raises(ConfigError, match="missing"):
        load_config("does_not_exist.yaml")
