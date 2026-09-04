"""Task 12 tests: Colab/Kaggle notebook wrappers and the operator runbook.

Every test here is offline and static: it parses committed notebook JSON and
Markdown with the standard library only. Nothing executes a notebook, installs a
package, touches the network, loads a model, or needs a GPU.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
COLAB_NOTEBOOK = NOTEBOOK_DIR / "colab_phase_ab.ipynb"
KAGGLE_NOTEBOOK = NOTEBOOK_DIR / "kaggle_phase_ab.ipynb"
RUNBOOK = REPO_ROOT / "docs" / "runbook-colab-kaggle.md"

NOTEBOOK_PATHS = {"colab": COLAB_NOTEBOOK, "kaggle": KAGGLE_NOTEBOOK}
PLATFORMS = tuple(NOTEBOOK_PATHS)

FULL_RUN_MARKER = "# AIF_SECTION: full_run"
SMOKE_MARKER = "# AIF_SECTION: smoke"

CANONICAL_CONFIG = "configs/phase_ab.yaml"
SMOKE_CONFIG = "configs/phase_ab_smoke.yaml"
GENERATED_CONFIG_DIR_MARKER = ".cache/aiforensics-notebook"

# The public Phase A/B command order. Assisted Qwen must follow CLIP probe
# because its Task 9 contract consumes CLIP assistant predictions.
CLI_SEQUENCE: tuple[str, ...] = (
    "aiforensics prepare",
    "aiforensics run --baseline clip_probe",
    "aiforensics run --baseline qwen_vl",
    "aiforensics run --baseline npr",
    "aiforensics run --baseline assisted_qwen",
    "aiforensics evaluate",
    "aiforensics report",
)

REQUIRED_STORAGE_ROOTS: tuple[str, ...] = (
    "DATA_ROOT",
    "MANIFEST_ROOT",
    "CACHE_ROOT",
    "OUTPUT_ROOT",
    "EXTERNAL_ROOT",
)

REQUIRED_CONFIG_ROOTS: tuple[str, ...] = (
    "data_root",
    "manifest_root",
    "cache_root",
    "output_root",
    "external_root",
)

RUNBOOK_SECTIONS: tuple[str, ...] = (
    "# Colab and Kaggle Runbook",
    "## Purpose",
    "## What Task 12 Does and Does Not Do",
    "## Prerequisites",
    "## Shared Phase A/B Command Order",
    "## Required Storage and Config Paths",
    "## Data and Manifest Provisioning",
    "## Google Colab",
    "## Kaggle",
    "## NPR Checkpoint and External Repository",
    "## GPU and Optional Dependencies",
    "## Smoke Verification",
    "## Full Phase A/B Run",
    "## Artifacts and Persistence",
    "## Failure / Deferred Troubleshooting",
    "## Reproducibility Checklist",
)

# Conservative secret detection: literal token values and literal assignments to
# credential-like names. Referencing an environment variable name is allowed.
SECRET_PATTERNS: tuple[str, ...] = (
    r"hf_[A-Za-z0-9]{20,}",
    r"sk-[A-Za-z0-9]{20,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"(?i)\b(?:api_key|apikey|secret|password|passwd|access_token|auth_token)"
    r"\s*=\s*[\"'][^\"']{8,}[\"']",
)

# Direct model-runtime imports belong in src/aiforensics/, never in a notebook.
FORBIDDEN_IMPORT_PATTERNS: tuple[str, ...] = (
    r"^\s*import\s+torch\b",
    r"^\s*from\s+torch\b",
    r"^\s*import\s+transformers\b",
    r"^\s*from\s+transformers\b",
    r"^\s*import\s+open_clip\b",
    r"^\s*from\s+open_clip\b",
)

FORBIDDEN_PACKAGE_IMPORT_PATTERNS: tuple[str, ...] = (
    r"^\s*(?:import|from)\s+aiforensics\b",
    r"aiforensics\.baselines",
    r"ClipProbeAdapter",
    r"QwenVLAdapter",
    r"AssistedQwenAdapter",
    r"NPRAdapter",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_notebook(path: Path) -> dict:
    assert path.is_file(), f"missing notebook: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]


def _markdown_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook.get("cells", []) if c.get("cell_type") == "markdown"]


def _code_source(notebook: dict) -> str:
    return "\n".join(_cell_source(c) for c in _code_cells(notebook))


def _all_source(notebook: dict) -> str:
    return "\n".join(_cell_source(c) for c in notebook.get("cells", []))


def _section_source(notebook: dict, marker: str) -> str:
    """Concatenate, in notebook order, code cells carrying a section marker."""
    return "\n".join(s for c in _code_cells(notebook) if marker in (s := _cell_source(c)))


@pytest.fixture(scope="module")
def notebooks() -> dict[str, dict]:
    return {name: _load_notebook(path) for name, path in NOTEBOOK_PATHS.items()}


@pytest.fixture(scope="module")
def runbook_text() -> str:
    assert RUNBOOK.is_file(), f"missing runbook: {RUNBOOK}"
    return RUNBOOK.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1-4. notebook structure and commit hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_exists_and_parses_as_json(platform: str) -> None:
    notebook = _load_notebook(NOTEBOOK_PATHS[platform])
    assert isinstance(notebook, dict)
    assert notebook.get("cells"), "notebook has no cells"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_declares_nbformat_4(platform: str, notebooks: dict[str, dict]) -> None:
    assert notebooks[platform].get("nbformat") == 4


@pytest.mark.parametrize("platform", PLATFORMS)
def test_code_cells_have_no_execution_count(platform: str, notebooks: dict[str, dict]) -> None:
    for index, cell in enumerate(_code_cells(notebooks[platform])):
        assert cell.get("execution_count", None) is None, f"cell {index} has an execution_count"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_code_cells_have_empty_outputs(platform: str, notebooks: dict[str, dict]) -> None:
    for index, cell in enumerate(_code_cells(notebooks[platform])):
        assert cell.get("outputs", []) == [], f"cell {index} has committed outputs"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_uses_python3_kernelspec(platform: str, notebooks: dict[str, dict]) -> None:
    kernelspec = notebooks[platform].get("metadata", {}).get("kernelspec", {})
    assert kernelspec.get("name") == "python3"


# ---------------------------------------------------------------------------
# 5-7. runtime config generation and storage inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_generates_runtime_config(platform: str, notebooks: dict[str, dict]) -> None:
    code = _code_source(notebooks[platform])
    assert CANONICAL_CONFIG in code, "canonical config must be read as a template"
    assert GENERATED_CONFIG_DIR_MARKER in code, "runtime config must be generated under .cache"
    assert f"phase_ab_{platform}.yaml" in code


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_never_writes_canonical_config(platform: str, notebooks: dict[str, dict]) -> None:
    """The committed configs must be templates only, never write targets."""
    for line in _code_source(notebooks[platform]).splitlines():
        if CANONICAL_CONFIG in line or SMOKE_CONFIG in line:
            assert '"w"' not in line and "'w'" not in line, f"in-place config write: {line!r}"
            assert "write_text" not in line, f"in-place config write: {line!r}"
            assert "safe_dump" not in line, f"in-place config write: {line!r}"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_exposes_storage_roots(platform: str, notebooks: dict[str, dict]) -> None:
    code = _code_source(notebooks[platform])
    for root in REQUIRED_STORAGE_ROOTS:
        assert re.search(rf"^\s*{root}\s*=", code, re.MULTILINE), f"{root} is not user-editable"
    for key in REQUIRED_CONFIG_ROOTS:
        assert key in code, f"config root {key} is never assigned in the runtime config"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_exposes_npr_checkpoint_path(platform: str, notebooks: dict[str, dict]) -> None:
    code = _code_source(notebooks[platform])
    assert re.search(r"^\s*NPR_CHECKPOINT_PATH\s*=", code, re.MULTILINE)
    assert "checkpoint_path" in code, "NPR checkpoint_path must be remapped in the runtime config"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_remaps_every_dataset_manifest(platform: str, notebooks: dict[str, dict]) -> None:
    code = _code_source(notebooks[platform])
    for field in ("train_manifest", "dev_manifest"):
        assert field in code
    assert code.count("manifest") >= 4


# ---------------------------------------------------------------------------
# 8-12. public CLI usage and ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_contains_prepare_command(platform: str, notebooks: dict[str, dict]) -> None:
    assert "aiforensics prepare" in _code_source(notebooks[platform])


@pytest.mark.parametrize("platform", PLATFORMS)
def test_full_run_has_exactly_one_clip_probe_command(
    platform: str, notebooks: dict[str, dict]
) -> None:
    """One CLI call owns every configured seed; no notebook-side seed loop."""
    full_run = _section_source(notebooks[platform], FULL_RUN_MARKER)
    assert full_run, "no full_run section marker found"
    assert full_run.count("--baseline clip_probe") == 1
    assert not re.search(r"for\s+seed\s+in", full_run), "notebook must not loop over seeds"


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("baseline", ["qwen_vl", "npr", "assisted_qwen"])
def test_full_run_contains_each_baseline_command(
    platform: str, baseline: str, notebooks: dict[str, dict]
) -> None:
    full_run = _section_source(notebooks[platform], FULL_RUN_MARKER)
    assert f"--baseline {baseline}" in full_run


@pytest.mark.parametrize("platform", PLATFORMS)
def test_full_run_contains_evaluate_and_report(platform: str, notebooks: dict[str, dict]) -> None:
    full_run = _section_source(notebooks[platform], FULL_RUN_MARKER)
    assert "aiforensics evaluate" in full_run
    assert "aiforensics report" in full_run


@pytest.mark.parametrize("platform", PLATFORMS)
def test_full_run_command_order_matches_contract(platform: str, notebooks: dict[str, dict]) -> None:
    full_run = _section_source(notebooks[platform], FULL_RUN_MARKER)
    positions = []
    for command in CLI_SEQUENCE:
        index = full_run.find(command)
        assert index != -1, f"missing command: {command}"
        positions.append(index)
    assert positions == sorted(positions), "full-run CLI order does not match the contract"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_does_not_suppress_cli_failures(platform: str, notebooks: dict[str, dict]) -> None:
    code = _code_source(notebooks[platform])
    assert "|| true" not in code
    assert "except Exception: pass" not in code
    assert not re.search(r"except\s+BaseException", code)
    for cell in _code_cells(notebooks[platform]):
        source = _cell_source(cell)
        if "aiforensics " in source and source.lstrip().startswith("%%bash"):
            assert "set -euo pipefail" in source, "CLI bash cells must fail loudly"


# ---------------------------------------------------------------------------
# 13-15. thin-wrapper and secret hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_does_not_import_package_internals(
    platform: str, notebooks: dict[str, dict]
) -> None:
    code = _code_source(notebooks[platform])
    for pattern in FORBIDDEN_PACKAGE_IMPORT_PATTERNS:
        assert not re.search(pattern, code, re.MULTILINE), f"notebook uses internals: {pattern}"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_does_not_import_model_runtimes(platform: str, notebooks: dict[str, dict]) -> None:
    code = _code_source(notebooks[platform])
    for pattern in FORBIDDEN_IMPORT_PATTERNS:
        assert not re.search(pattern, code, re.MULTILINE), f"notebook imports runtime: {pattern}"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_contains_no_embedded_secrets(platform: str, notebooks: dict[str, dict]) -> None:
    source = _all_source(notebooks[platform])
    for pattern in SECRET_PATTERNS:
        assert not re.search(pattern, source), f"possible embedded secret: {pattern}"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_contains_no_personal_absolute_paths(
    platform: str, notebooks: dict[str, dict]
) -> None:
    source = _all_source(notebooks[platform])
    for pattern in (
        r"/Users/[A-Za-z0-9._-]+",
        r"/home/(?!jovyan\b)[A-Za-z0-9._-]+",
        r"C:\\\\Users",
    ):
        assert not re.search(pattern, source), f"personal absolute path: {pattern}"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_has_markdown_before_first_code_cell(
    platform: str, notebooks: dict[str, dict]
) -> None:
    cells = notebooks[platform].get("cells", [])
    first_code = next(i for i, c in enumerate(cells) if c.get("cell_type") == "code")
    assert any(c.get("cell_type") == "markdown" for c in cells[:first_code])
    assert len(_markdown_cells(notebooks[platform])) >= 8


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_carries_no_stale_execution_metadata(
    platform: str, notebooks: dict[str, dict]
) -> None:
    notebook = notebooks[platform]
    assert "widgets" not in notebook.get("metadata", {})
    for index, cell in enumerate(notebook.get("cells", [])):
        metadata = cell.get("metadata", {})
        assert "execution" not in metadata, f"cell {index} carries execution timings"
        assert "outputId" not in metadata, f"cell {index} carries a saved output id"


# ---------------------------------------------------------------------------
# 16-17. platform-specific guidance
# ---------------------------------------------------------------------------


def test_colab_notebook_has_drive_mount_guidance(notebooks: dict[str, dict]) -> None:
    notebook = notebooks["colab"]
    code = _code_source(notebook)
    prose = _all_source(notebook)
    assert "from google.colab import drive" in code
    assert re.search(r"^\s*DRIVE_MOUNT_POINT\s*=", code, re.MULTILINE)
    assert "drive.mount" in code
    assert "ephemeral" in prose.lower()


def test_kaggle_notebook_separates_readonly_input_from_writable_output(
    notebooks: dict[str, dict],
) -> None:
    notebook = notebooks["kaggle"]
    code = _code_source(notebook)
    prose = _all_source(notebook).lower()
    assert re.search(r"^\s*KAGGLE_INPUT_ROOT\s*=", code, re.MULTILINE)
    assert re.search(r"^\s*KAGGLE_WORKING_ROOT\s*=", code, re.MULTILINE)
    assert "/kaggle/input" in code
    assert "/kaggle/working" in code
    assert "read-only" in prose
    assert "writable" in prose


def test_kaggle_notebook_hardcodes_no_dataset_slug(notebooks: dict[str, dict]) -> None:
    code = _code_source(notebooks["kaggle"])
    for line in code.splitlines():
        if "/kaggle/input/" in line:
            assert "<" in line and ">" in line, f"hardcoded dataset path: {line!r}"


# ---------------------------------------------------------------------------
# manifest building contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_exposes_build_manifests_switch(platform: str, notebooks: dict[str, dict]) -> None:
    """Building overwrites manifest CSVs, so it must be an explicit user choice."""
    code = _code_source(notebooks[platform])
    assert re.search(r"^\s*BUILD_MANIFESTS\s*=\s*(?:True|False)", code, re.MULTILINE)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_can_persist_hf_cache(platform: str, notebooks: dict[str, dict]) -> None:
    """Model weights survive session resets only if HF_HOME is redirected."""
    code = _code_source(notebooks[platform])
    assert re.search(r"^\s*PERSIST_HF_CACHE\s*=\s*(?:True|False)", code, re.MULTILINE)
    # The %%bash run cells must inherit the setting, so it goes through os.environ.
    assert 'os.environ["HF_HOME"]' in code
    # The cache must live in persistent storage, not the ephemeral default.
    assert "hf-cache" in code


@pytest.mark.parametrize("platform", PLATFORMS)
def test_prepare_build_flag_is_driven_by_the_switch_not_hardcoded(
    platform: str, notebooks: dict[str, dict]
) -> None:
    """The prepare command reads one exported variable instead of two code paths."""
    full_run = _section_source(notebooks[platform], FULL_RUN_MARKER)
    assert 'os.environ["AIF_PREPARE_ARGS"]' in full_run
    assert "aiforensics prepare ${AIF_PREPARE_ARGS}" in full_run
    assert "--build-manifests" in full_run
    # No shell cell may hardcode the flag: that would ignore BUILD_MANIFESTS.
    for cell in _code_cells(notebooks[platform]):
        source = _cell_source(cell)
        if source.lstrip().startswith("%%bash"):
            assert "prepare --build-manifests" not in source


@pytest.mark.parametrize("platform", PLATFORMS)
def test_prepare_still_runs_before_every_baseline(
    platform: str, notebooks: dict[str, dict]
) -> None:
    full_run = _section_source(notebooks[platform], FULL_RUN_MARKER)
    prepare_index = full_run.find("aiforensics prepare")
    assert prepare_index != -1
    for baseline in ("clip_probe", "qwen_vl", "npr", "assisted_qwen"):
        assert prepare_index < full_run.find(f"--baseline {baseline}")


def test_kaggle_manifest_root_is_writable_when_building(notebooks: dict[str, dict]) -> None:
    """/kaggle/input is read-only, so a built manifest cannot live there."""
    code = _code_source(notebooks["kaggle"])
    match = re.search(r"^\s*MANIFEST_ROOT\s*=\s*(.+)$", code, re.MULTILINE)
    assert match, "MANIFEST_ROOT assignment not found"
    assignment = match.group(1)
    assert "BUILD_MANIFESTS" in assignment, "manifest root must depend on the build switch"
    assert "KAGGLE_WORKING_ROOT" in assignment, "built manifests need writable storage"


def test_kaggle_creates_manifest_root_only_when_building(notebooks: dict[str, dict]) -> None:
    code = _code_source(notebooks["kaggle"])
    assert "writable_roots" in code
    assert re.search(r"if\s+BUILD_MANIFESTS:\s*\n\s*writable_roots\.append", code)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_reports_discovered_generator_directories(
    platform: str, notebooks: dict[str, dict]
) -> None:
    """A wrong DATA_ROOT is the most common first-run mistake; surface it early."""
    code = _code_source(notebooks[platform])
    assert "found_generators" in code
    assert "configured_generators" in code
    assert "missing_generators" in code


@pytest.mark.parametrize("platform", PLATFORMS)
def test_generator_discovery_matches_the_builder_layout(
    platform: str, notebooks: dict[str, dict]
) -> None:
    """The notebook's preflight must use the same layout rule as the builder."""
    from aiforensics.data.genimage import _LABEL_DIRS, _SPLIT_DIRS

    code = _code_source(notebooks[platform])
    for split_dir in _SPLIT_DIRS:
        assert f'"{split_dir}"' in code, f"notebook does not check the {split_dir} directory"
    for label_dir in _LABEL_DIRS:
        assert label_dir in code, f"notebook does not mention the {label_dir} directory"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_does_not_reimplement_manifest_building(
    platform: str, notebooks: dict[str, dict]
) -> None:
    """Discovery may list directories; checksums and CSV writing stay in the package."""
    code = _code_source(notebooks[platform])
    assert "sha256" not in code.lower()
    assert "import csv" not in code
    assert "write_manifest" not in code
    assert "ManifestRecord" not in code


# ---------------------------------------------------------------------------
# smoke section separation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_smoke_section_is_separate_and_uses_smoke_config(
    platform: str, notebooks: dict[str, dict]
) -> None:
    smoke = _section_source(notebooks[platform], SMOKE_MARKER)
    assert smoke, "no smoke section marker found"
    assert SMOKE_CONFIG in smoke
    full_run = _section_source(notebooks[platform], FULL_RUN_MARKER)
    assert SMOKE_CONFIG not in full_run, "smoke config must not appear in the full run"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_smoke_section_keeps_non_scientific_warning(
    platform: str, notebooks: dict[str, dict]
) -> None:
    prose = _all_source(notebooks[platform]).lower()
    assert "not scientific evidence" in prose


@pytest.mark.parametrize("platform", PLATFORMS)
def test_smoke_runtime_config_only_relocates_cache_and_output(
    platform: str, notebooks: dict[str, dict]
) -> None:
    """Smoke data/manifests must keep pointing at the repository fixtures."""
    smoke = _section_source(notebooks[platform], SMOKE_MARKER)
    assert 'paths"]["cache_root' in smoke or "paths']['cache_root" in smoke
    assert 'paths"]["output_root' in smoke or "paths']['output_root" in smoke
    assert 'paths"]["data_root' not in smoke, "smoke data_root must stay on repo fixtures"
    assert "train_manifest" not in smoke, "smoke manifests must stay on repo fixtures"


# ---------------------------------------------------------------------------
# Python 3.10 provisioning contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_provisions_python310_without_patching_metadata(
    platform: str, notebooks: dict[str, dict]
) -> None:
    code = _code_source(notebooks[platform])
    assert "3.10" in code
    # uv is invoked in subprocess list form, so check the semantic arguments.
    assert '"venv"' in code or "uv venv" in code
    assert "--seed" in code, "the provisioned venv needs pip seeded for the install cell"
    assert "PATH" in code, "the provisioned environment must be prepended to PATH"
    assert "pyproject.toml" in code, "repository layout must be validated"
    for line in code.splitlines():
        if "pyproject.toml" in line:
            assert "write_text" not in line, "must not patch pyproject.toml"
            assert '"w"' not in line and "'w'" not in line, "must not patch pyproject.toml"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_install_uses_repository_extras(platform: str, notebooks: dict[str, dict]) -> None:
    code = _code_source(notebooks[platform])
    assert "pip install" in code
    assert "-e" in code
    assert "[clip,qwen,npr]" in code
    assert "==" not in code.split("pip install")[1][:200], "do not duplicate pinned versions"


# ---------------------------------------------------------------------------
# 18-23. runbook contract
# ---------------------------------------------------------------------------


def test_runbook_sections_present_and_ordered(runbook_text: str) -> None:
    positions = []
    for heading in RUNBOOK_SECTIONS:
        index = runbook_text.find(heading)
        assert index != -1, f"missing runbook section: {heading}"
        positions.append(index)
    assert positions == sorted(positions), "runbook sections are out of order"


def test_runbook_documents_all_five_roots(runbook_text: str) -> None:
    for root in REQUIRED_CONFIG_ROOTS:
        assert root in runbook_text, f"runbook does not document {root}"
    for variable in REQUIRED_STORAGE_ROOTS:
        assert variable in runbook_text, f"runbook does not name the {variable} variable"


def test_runbook_documents_manifest_and_checkpoint_paths(runbook_text: str) -> None:
    for name in (
        "tiny_genimage_train.csv",
        "tiny_genimage_dev.csv",
        "genimage_unseen_external.csv",
        "synthbuster_external.csv",
        "NPR_CHECKPOINT_PATH",
    ):
        assert name in runbook_text


def test_runbook_states_prepare_does_not_provision_datasets(runbook_text: str) -> None:
    lowered = runbook_text.lower()
    assert "not a research-dataset downloader" in lowered or "not implemented" in lowered
    assert "prepare" in lowered
    assert "validates" in lowered


def test_runbook_documents_manifest_building(runbook_text: str) -> None:
    """The build mode, its layout contract, and its leakage rules must be written down."""
    lowered = runbook_text.lower()
    assert "--build-manifests" in lowered
    assert "build_manifests" in lowered
    for token in ("<generator>", "ai", "nature", "train", "val"):
        assert token in lowered, f"runbook does not document the {token} layout element"
    assert "per-generator" in lowered, "the cap semantics must be explicit"
    assert "sha-256" in lowered or "sha256" in lowered
    assert "shortcut" in lowered, "format-skew guidance must be documented"


def test_runbook_documents_writable_manifest_root_for_building(runbook_text: str) -> None:
    """/kaggle/input is read-only, so the build-mode storage split must be stated."""
    assert "MANIFEST_ROOT" in runbook_text
    assert "/kaggle/working/manifests" in runbook_text


def test_runbook_documents_npr_checkpoint_and_external_repo(runbook_text: str) -> None:
    lowered = runbook_text.lower()
    assert "external_root" in lowered
    assert "checkpoint" in lowered
    assert "sha-256" in lowered or "sha256" in lowered
    assert "official" in lowered
    assert "pinned" in lowered


def test_runbook_marks_smoke_as_non_scientific(runbook_text: str) -> None:
    assert "not scientific evidence" in runbook_text.lower()


def test_runbook_contains_shared_cli_sequence(runbook_text: str) -> None:
    positions = []
    for command in CLI_SEQUENCE:
        index = runbook_text.find(command)
        assert index != -1, f"runbook is missing command: {command}"
        positions.append(index)
    assert positions == sorted(positions), "runbook CLI order does not match the contract"
    assert "assisted_qwen" in runbook_text
    assert "clip" in runbook_text.lower()


def test_runbook_documents_deferred_versus_failed(runbook_text: str) -> None:
    lowered = runbook_text.lower()
    assert "deferred" in lowered
    assert "failed" in lowered
    assert "`deferred`" in lowered or "deferred versus failed" in lowered


def test_runbook_documents_python310_provisioning_and_network_needs(runbook_text: str) -> None:
    lowered = runbook_text.lower()
    assert "3.10" in lowered
    assert "uv venv" in lowered
    assert "internet" in lowered, "Kaggle Internet requirement must be documented"
    assert "network" in lowered


def test_runbook_troubleshooting_covers_required_cases(runbook_text: str) -> None:
    lowered = runbook_text.lower()
    for topic in (
        "python",
        "gpu",
        "qwen",
        "manifest",
        "checkpoint",
        "network",
        "checksum",
        "read-only",
        "session",
        "deferred",
        "incomplete",
    ):
        assert topic in lowered, f"troubleshooting does not mention {topic}"
    # `|| true` may only appear as a prohibition, never as advice.
    for line in runbook_text.splitlines():
        if "|| true" in line:
            assert re.search(r"\b(?:do not|don't|never|must not)\b", line, re.IGNORECASE), (
                f"runbook appears to recommend suppressing failures: {line!r}"
            )


def test_runbook_does_not_embed_personal_paths_or_secrets(runbook_text: str) -> None:
    for pattern in (r"/Users/[A-Za-z0-9._-]+", *SECRET_PATTERNS):
        assert not re.search(pattern, runbook_text), f"runbook leaks: {pattern}"
