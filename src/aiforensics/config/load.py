import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from aiforensics.config.models import AppConfig


class ConfigError(ValueError):
    pass


def find_repo_root(start_path: Path) -> Path:
    current = start_path.resolve()
    if current.is_file():
        current = current.parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise ConfigError(f"Could not find repository root (pyproject.toml) starting from {start_path}")


def _resolve_path_value(val: Any, root_dir: Path) -> Any:
    """Resolve a single path representation against root_dir if relative."""
    if val is None:
        return val
    p = Path(str(val))
    return str(p) if p.is_absolute() else str(root_dir / p)


def resolve_paths(config_dict: dict[str, Any], root_dir: Path) -> dict[str, Any]:
    """Returns a new config dictionary with relative paths resolved to absolute paths."""
    resolved = copy.deepcopy(config_dict)

    path_fields = [
        (("paths", "data_root"),),
        (("paths", "manifest_root"),),
        (("paths", "cache_root"),),
        (("paths", "output_root"),),
        (("paths", "external_root"),),
        (("datasets", "tiny_genimage", "train_manifest"),),
        (("datasets", "tiny_genimage", "dev_manifest"),),
        (("datasets", "genimage_unseen", "manifest"),),
        (("datasets", "synthbuster", "manifest"),),
        (("baselines", "npr", "checkpoint_path"),),
    ]

    for key_path_tuple in path_fields:
        keys = key_path_tuple[0]
        curr = resolved
        for k in keys[:-1]:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                curr = None
                break

        if curr is not None and isinstance(curr, dict) and keys[-1] in curr:
            curr[keys[-1]] = _resolve_path_value(curr[keys[-1]], root_dir)

    return resolved


def load_config(path: Path | str) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file missing: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML structure in {config_path}: {exc}") from exc

    if not isinstance(config_dict, dict):
        raise ConfigError(
            f"Config file {config_path} does not contain a valid dictionary structure"
        )

    required_sections = [
        "project",
        "paths",
        "runtime",
        "datasets",
        "baselines",
        "evaluation",
        "report",
    ]
    for section in required_sections:
        if section not in config_dict:
            raise ConfigError(f"Missing required section in {config_path}: {section}")

    repo_root = find_repo_root(config_path)
    resolved_dict = resolve_paths(config_dict, repo_root)

    try:
        return AppConfig(**resolved_dict)
    except ValidationError as exc:
        error_msgs = []
        for err in exc.errors():
            loc_str = ".".join([str(loc) for loc in err["loc"]])
            error_msgs.append(f"Field '{loc_str}': {err['msg']}")

        raise ConfigError(
            f"Config validation error in {config_path}: " + " | ".join(error_msgs)
        ) from exc
