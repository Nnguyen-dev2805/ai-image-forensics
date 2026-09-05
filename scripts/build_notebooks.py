"""One-shot generator for the Task 12 hosted-notebook wrappers.

Run from the repository root:

    uv run python scripts/build_notebooks.py

Notebook JSON is generated rather than hand-edited so that committed cells stay
free of execution counts, outputs, and stale metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# shared cell bodies
# ---------------------------------------------------------------------------

TITLE_COLAB = """
# Phase A/B on Google Colab

This notebook is a **thin wrapper** around the `aiforensics` command-line
interface that already lives in this repository. It installs the package,
prepares an environment, points the pipeline at storage you control, and then
calls the same public CLI you would run locally.

It does **not** implement dataset parsing, manifest validation, model loading,
inference, metrics, reporting, caching, or NPR checkout. Those behaviours belong
to the package (Tasks 1-11) and must stay there.

Before you start, understand the limits:

- A full Phase A/B comparison needs **pre-provisioned images and manifests**.
  `aiforensics prepare` validates what already exists; it is not a
  research-dataset downloader.
- The heavy baselines (Qwen-VL, Assisted Qwen, NPR) generally need **CUDA**,
  network access for model weights, and an operator-provided NPR checkpoint.
- The optional smoke section proves the pipeline works in this environment.
  Smoke metrics are pipeline checks and **not scientific evidence**.
- Colab runtime storage is **ephemeral**: anything written outside mounted Drive
  disappears when the runtime is recycled.

See `docs/runbook-colab-kaggle.md` for the operator runbook.
"""

TITLE_KAGGLE = """
# Phase A/B on Kaggle

This notebook is a **thin wrapper** around the `aiforensics` command-line
interface that already lives in this repository. It installs the package,
prepares an environment, points the pipeline at storage you control, and then
calls the same public CLI you would run locally.

It does **not** implement dataset parsing, manifest validation, model loading,
inference, metrics, reporting, caching, or NPR checkout. Those behaviours belong
to the package (Tasks 1-11) and must stay there.

Before you start, understand the limits:

- A full Phase A/B comparison needs **pre-provisioned images and manifests**.
  `aiforensics prepare` validates what already exists; it is not a
  research-dataset downloader.
- The heavy baselines (Qwen-VL, Assisted Qwen, NPR) generally need **CUDA**,
  network access for model weights, and an operator-provided NPR checkpoint.
- The optional smoke section proves the pipeline works in this environment.
  Smoke metrics are pipeline checks and **not scientific evidence**.
- Attached datasets under `/kaggle/input` are **read-only**. Every generated
  artifact must be written to writable storage such as `/kaggle/working`, which
  is also **ephemeral** once the session ends unless you save the output.
- Installing packages and provisioning Python 3.10 needs the notebook
  **Internet** setting to be enabled.

See `docs/runbook-colab-kaggle.md` for the operator runbook.
"""

TITLE_KAGGLE_VERTEX_QWEN = """
# Phase A/B on Kaggle with Vertex Qwen

This notebook is the quick end-to-end path for validating
`Qwen/Qwen2.5-VL-7B-Instruct` through a Google Cloud Vertex AI Dedicated
Endpoint and including that result in the normal `aiforensics` report.

CLIP and NPR stay on the existing public CLI path. Qwen is called here through
the same public CLI, using the OpenAI-compatible endpoint with Kaggle Secrets
based authentication.
"""

TITLE_KAGGLE_VERTEX_QWEN_ALL_VAL = """
# Qwen-VL All-Val Evaluation on Kaggle (via Vertex AI)

This notebook is a **thin wrapper** around the `aiforensics` CLI to evaluate the
`qwen_vl` baseline via a Google Cloud Vertex AI Dedicated Endpoint across **all**
generators found under `DATA_ROOT` using **only** the dataset `val` split.

It does **not** train any model, and does **not** run CLIP, NPR, or Assisted Qwen.

Key features:
- Reads credentials securely from the Kaggle Secret `GOOGLE_APPLICATION_CREDENTIALS`.
- Uses Dedicated Endpoint domain `*.prediction.vertexai.goog`.
- Discovers all generator directories under `<DATA_ROOT>/<generator>/val/{ai,nature}/*`.
- Configurable `MAX_IMAGES_PER_GENERATOR = 0` (0 evaluates all available val images).
- Writes inspectable artifacts, metrics, and report under `/kaggle/working/outputs-qwen-all-val/`.
"""

PREFLIGHT_MD = """
## 1. Runtime preflight

This cell only reports what the environment looks like. It deliberately does
**not** fail when the notebook kernel is newer than Python 3.10: the kernel never
imports `aiforensics`, the CLI does. What matters is whether a Python 3.10
interpreter is available for the CLI, because `pyproject.toml` declares

```text
requires-python = ">=3.10,<3.11"
```

GPU output below is informational. Real device selection and deferral stay
inside the baseline adapters.
"""

PREFLIGHT_CODE = '''
import os
import shutil
import subprocess
import sys
from pathlib import Path

# User-editable: where the Python 3.10 environment for the CLI lives.
CLI_VENV_PATH = Path("/content/aiforensics-venv310")

TARGET_PY = (3, 10)


def _venv_bin(venv_path: Path) -> Path:
    """Return the scripts directory of a virtual environment."""
    return venv_path / ("Scripts" if os.name == "nt" else "bin")


def _interpreter_version(executable: str) -> tuple[int, int] | None:
    """Return (major, minor) for an interpreter, or None when unusable."""
    try:
        result = subprocess.run(
            [executable, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    parts = result.stdout.split()
    if len(parts) != 2:
        return None
    return int(parts[0]), int(parts[1])


def find_cli_python() -> str | None:
    """Find an interpreter that satisfies the repository Python contract."""
    candidates = [
        str(_venv_bin(CLI_VENV_PATH) / "python"),
        shutil.which("python3.10"),
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        if _interpreter_version(candidate) == TARGET_PY:
            return candidate
    return None


print("notebook kernel:", sys.version.split()[0], "(informational only)")
print("working directory:", Path.cwd())

CLI_PYTHON = find_cli_python()
if CLI_PYTHON:
    print("Python 3.10 for the CLI:", CLI_PYTHON)
else:
    print(
        "No Python 3.10 interpreter found yet.\\n"
        "Run the optional provisioning cell in section 2 before installing the package."
    )

gpu = shutil.which("nvidia-smi")
if gpu:
    subprocess.run([gpu], check=False)
else:
    print("nvidia-smi not found: no GPU visible to this runtime (informational).")
'''

PROVISION_MD = """
## 2. Optional: provision Python 3.10 for the CLI

Run this section **only when section 1 reported no Python 3.10 interpreter**.

It creates a dedicated virtual environment on Python 3.10 and puts it first on
`PATH`, so later cells can call `aiforensics` unchanged. This satisfies the
repository's `requires-python` contract honestly: the package is installed under
a real 3.10 interpreter. Never edit `pyproject.toml` to make an install succeed.

This step needs **network access** (to fetch `uv`, the interpreter, and the
dependencies).
"""

PROVISION_CODE = '''
def provision_cli_python(venv_path: Path) -> str:
    """Create a Python 3.10 virtual environment and prepend it to PATH."""
    if shutil.which("uv") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "uv"],
            check=True,
        )

    uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    subprocess.run([uv, "python", "install", "3.10"], check=True)
    subprocess.run(
        [uv, "venv", "--seed", "--no-project", "--python", "3.10", str(venv_path)],
        check=True,
    )

    bin_dir = _venv_bin(venv_path)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["VIRTUAL_ENV"] = str(venv_path)
    return str(bin_dir / "python")


CLI_PYTHON = provision_cli_python(CLI_VENV_PATH)
print("provisioned CLI interpreter:", CLI_PYTHON)
'''

VERIFY_PY_MD = """
### 2b. Verify the CLI interpreter

This is the first hard gate. If no valid Python 3.10 environment exists after
provisioning, stop here and fix the environment instead of working around the
version contract.
"""

VERIFY_PY_CODE = """
resolved = shutil.which("python") or ""
version = _interpreter_version(resolved) if resolved else None

print("python on PATH:", resolved or "<none>")
print("python version:", ".".join(str(p) for p in version) if version else "<unknown>")

if version != TARGET_PY:
    raise RuntimeError(
        "No usable Python 3.10 environment for the CLI. The repository requires "
        ">=3.10,<3.11. Run the provisioning cell above, or select a runtime that "
        "provides Python 3.10. Do not modify pyproject.toml to bypass this."
    )

print("OK: the CLI will run under Python 3.10.")
"""

REPO_MD_COLAB = """
## 3. Repository location

Point `REPO_ROOT` at a checkout of this repository. Either use a clone that is
already present in the runtime, or set `REPO_GIT_URL` to a repository you control
and let the cell clone it. Do not embed credentials in this notebook; use a
Colab secret or an environment variable if a private clone needs authentication.
"""

REPO_MD_KAGGLE = """
## 3. Repository location

Point `REPO_ROOT` at a checkout of this repository. Either attach it as a Kaggle
dataset/utility script and copy it into writable storage, or set `REPO_GIT_URL`
to a repository you control and let the cell clone it (needs Internet enabled).
Do not embed credentials in this notebook; use an environment variable or Kaggle
Secrets if a private clone needs authentication.
"""

REPO_CODE_TEMPLATE = """
# User-editable inputs. The defaults below are this project's own values so a
# fresh session needs no editing; override them for a fork or mirror.
REPO_ROOT = Path("{repo_root}")
REPO_GIT_URL = "https://github.com/Nnguyen-dev2805/ai-image-forensics.git"

if not REPO_ROOT.exists() and REPO_GIT_URL:
    REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO_GIT_URL, str(REPO_ROOT)], check=True)

REQUIRED_REPO_FILES = [
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "configs/phase_ab.yaml",
]
missing = [str(path) for path in REQUIRED_REPO_FILES if not path.is_file()]
if missing:
    raise FileNotFoundError(
        "REPO_ROOT does not look like this repository. Missing: "
        + ", ".join(missing)
        + ". Set REPO_ROOT (or REPO_GIT_URL) to a valid checkout."
    )

os.chdir(REPO_ROOT)
print("repository root:", REPO_ROOT)
"""

INSTALL_MD = """
## 4. Install the package

Dependency names come from `pyproject.toml`; nothing is pinned again here. The
optional extras map to the baselines: `clip` for the CLIP probe, `qwen` for
Qwen-VL and Assisted Qwen, `npr` for the NPR runtime bridge.

Model weights and the NPR checkpoint are **not** packaged with the repository.
"""

INSTALL_CODE = """
%%bash
set -euo pipefail

cd "$AIF_REPO_ROOT"
python -m pip install --quiet --upgrade pip
python -m pip install -e ".[clip,qwen,npr]"
"""

INSTALL_VERTEX_CODE = """
import importlib
import shutil
import subprocess
import sys


def vertex_dependencies_ready() -> bool:
    try:
        importlib.import_module("kaggle_secrets")
        importlib.import_module("requests")
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        from openai import OpenAI
    except ImportError:
        return False
    _ = (OpenAI, Request, service_account)
    return shutil.which("aiforensics") is not None


if vertex_dependencies_ready():
    print("Vertex dependencies already import; skipping pip install")
else:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[clip,vertex,npr]"],
        cwd=str(REPO_ROOT),
        check=True,
    )
"""

INSTALL_VERTEX_ALL_VAL_CODE = """
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

# 1. Locate the Python 3.10 CLI virtual environment provisioned in Section 2
cli_python = shutil.which("python")
cli_py_version = (
    subprocess.run([cli_python, "-V"], capture_output=True, text=True).stdout
    if cli_python
    else ""
)
if not cli_python or "3.10" not in cli_py_version:
    cli_python = str(Path("/kaggle/working/aiforensics-venv310/bin/python"))

print("CLI Python 3.10 target:", cli_python)

# 2. Install aiforensics with [vertex] extra into the Python 3.10 environment
subprocess.run(
    [cli_python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
    check=True,
)
subprocess.run(
    [cli_python, "-m", "pip", "install", "-e", ".[vertex]"],
    cwd=str(REPO_ROOT),
    check=True,
)

# 3. Ensure the notebook kernel (Python 3.12) has openai and google dependencies for Section 8
try:
    import openai
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "openai", "google-auth", "requests"],
        check=True,
    )

print("aiforensics CLI and Vertex dependencies installed successfully!")
"""

INSTALL_ENV_CODE = """
os.environ["AIF_REPO_ROOT"] = str(REPO_ROOT)
print("AIF_REPO_ROOT =", os.environ["AIF_REPO_ROOT"])
"""

VERIFY_CLI_MD = """
### 4b. Verify the CLI resolves from the Python 3.10 environment

Second hard gate: the `aiforensics` entry point must exist and run under the
interpreter verified in section 2b.
"""

VERIFY_CLI_CODE = """
cli_path = shutil.which("aiforensics")
print("aiforensics on PATH:", cli_path or "<none>")

if not cli_path:
    raise RuntimeError(
        "The aiforensics CLI is not on PATH. Re-run the install cell, and make "
        "sure the Python 3.10 environment from section 2 is still first on PATH."
    )

result = subprocess.run([cli_path, "--help"], capture_output=True, text=True, check=False)
if result.returncode != 0:
    raise RuntimeError(f"aiforensics --help failed with exit code {result.returncode}")

print("OK: CLI available at", cli_path)
"""

STORAGE_MD_COLAB = """
## 5. Storage inputs

Every project root is an explicit, user-editable variable. Choose between:

1. **ephemeral runtime storage** (fast, disappears when the runtime recycles),
2. **mounted Drive storage** (persists caches, outputs, external checkout,
   checkpoint, and/or research data across sessions).

`DRIVE_MOUNT_POINT` and `PERSIST_ROOT` are examples you are expected to change.
Nothing here assumes a particular Drive folder name.
"""

STORAGE_CODE_COLAB = """
USE_DRIVE = False  # set True to keep artifacts on Google Drive

# User-editable: Colab mount point and the folder you want to use inside Drive.
DRIVE_MOUNT_POINT = Path("/content/drive")
DRIVE_PROJECT_SUBPATH = "MyDrive/<your-folder>/ai-image-forensics"  # example: change this

if USE_DRIVE:
    from google.colab import drive

    drive.mount(str(DRIVE_MOUNT_POINT))
    PERSIST_ROOT = DRIVE_MOUNT_POINT / DRIVE_PROJECT_SUBPATH
else:
    PERSIST_ROOT = Path("/content/aiforensics-workspace")

# The five project roots required by the pipeline, plus explicit manifest and
# checkpoint paths. Point them anywhere you control.
DATA_ROOT = PERSIST_ROOT / "data"
MANIFEST_ROOT = PERSIST_ROOT / "manifests"
CACHE_ROOT = PERSIST_ROOT / "cache"
OUTPUT_ROOT = PERSIST_ROOT / "outputs"
EXTERNAL_ROOT = PERSIST_ROOT / "external"
NPR_CHECKPOINT_PATH = PERSIST_ROOT / "checkpoints" / "NPR.pth"

# Set True to build manifests from a GenImage-layout DATA_ROOT
# (<generator>/<train|val>/<ai|nature>/) in this session; set False when the
# manifest CSVs below already exist. Building overwrites those CSV files.
BUILD_MANIFESTS = True

# Set True to keep HuggingFace model weights (~6 GB for Qwen) inside persistent
# storage, so a saved notebook version restores them without re-downloading.
# Costs output quota and makes "Save Version" slower. False keeps the default
# ephemeral cache, which is fine when downloads are cheap.
PERSIST_HF_CACHE = True
if PERSIST_HF_CACHE:
    HF_CACHE_ROOT = PERSIST_ROOT / "hf-cache"
    HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    # The %%bash run cells inherit this, so the CLI's transformers downloads
    # land in persistent storage too.
    os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
else:
    HF_CACHE_ROOT = None

# Manifest filenames are separate config fields today; override them if your
# provisioned manifests use different names.
TINY_TRAIN_MANIFEST = MANIFEST_ROOT / "tiny_genimage_train.csv"
TINY_DEV_MANIFEST = MANIFEST_ROOT / "tiny_genimage_dev.csv"
GENIMAGE_UNSEEN_MANIFEST = MANIFEST_ROOT / "genimage_unseen_external.csv"
SYNTHBUSTER_MANIFEST = MANIFEST_ROOT / "synthbuster_external.csv"

for writable in (CACHE_ROOT, OUTPUT_ROOT, EXTERNAL_ROOT, MANIFEST_ROOT):
    writable.mkdir(parents=True, exist_ok=True)

print("persist root :", PERSIST_ROOT)
print("data root    :", DATA_ROOT)
print("manifest root:", MANIFEST_ROOT)
print("cache root   :", CACHE_ROOT)
print("output root  :", OUTPUT_ROOT)
print("external root:", EXTERNAL_ROOT)
print("npr ckpt     :", NPR_CHECKPOINT_PATH)
print("hf cache     :", HF_CACHE_ROOT if HF_CACHE_ROOT else "(default, ephemeral)")
print("build manifests:", BUILD_MANIFESTS)
"""

STORAGE_MD_KAGGLE = """
## 5. Storage inputs

Kaggle separates **read-only** attached data from **writable** working storage:

- `/kaggle/input/<your-dataset>` is read-only. Research images and pre-built
  manifests normally live here.
- `/kaggle/working` is writable but ephemeral; save the notebook output to keep
  anything beyond the session.

No dataset slug or username is hardcoded. Replace the `<...>` placeholders with
the datasets you attached.
"""

STORAGE_CODE_KAGGLE = """
# User-editable: Kaggle mount points.
KAGGLE_INPUT_ROOT = Path("/kaggle/input")  # read-only attached datasets
KAGGLE_WORKING_ROOT = Path("/kaggle/working")  # writable, ephemeral

# Full path to the attached dataset directory that holds the images. Kaggle
# mounts datasets under different shapes (/kaggle/input/<slug> and
# /kaggle/input/datasets/<owner>/<slug> both occur), so give the whole path
# instead of assuming one layout. Prefilled with this project's dataset.
INPUT_DATA_DIR = KAGGLE_INPUT_ROOT / "datasets/yangsangtai/tiny-genimage"
INPUT_CHECKPOINT_DIR = KAGGLE_INPUT_ROOT / "<your-npr-checkpoint-dataset>"

# Set True to build manifests from a GenImage-layout INPUT_DATA_DIR in this
# session; set False when manifest CSVs are already provisioned in an attached
# dataset. Building writes CSV files, and /kaggle/input is read-only, so the two
# modes cannot share the same manifest root.
BUILD_MANIFESTS = True
INPUT_MANIFEST_DIR = KAGGLE_INPUT_ROOT / "<your-manifests-dataset>"

# Set True to keep HuggingFace model weights (~6 GB for Qwen) inside
# /kaggle/working, so saved notebook output restores them without a re-download
# after a session reset. Costs output quota (~6 GB of ~20 GB) and makes saving
# slower. False keeps the default ephemeral cache under /root/.cache.
PERSIST_HF_CACHE = True
if PERSIST_HF_CACHE:
    HF_CACHE_ROOT = KAGGLE_WORKING_ROOT / "hf-cache"
    HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    # The %%bash run cells inherit this, so the CLI's transformers downloads
    # land in persistent storage too.
    os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
else:
    HF_CACHE_ROOT = None

# Read-only inputs.
DATA_ROOT = INPUT_DATA_DIR
NPR_CHECKPOINT_PATH = INPUT_CHECKPOINT_DIR / "NPR.pth"

# Writable outputs: never place these under /kaggle/input.
CACHE_ROOT = KAGGLE_WORKING_ROOT / "cache"
OUTPUT_ROOT = KAGGLE_WORKING_ROOT / "outputs"
EXTERNAL_ROOT = KAGGLE_WORKING_ROOT / "external"

# Built manifests must land in writable storage; provisioned ones stay read-only.
MANIFEST_ROOT = KAGGLE_WORKING_ROOT / "manifests" if BUILD_MANIFESTS else INPUT_MANIFEST_DIR

# Manifest filenames are separate config fields today; override them if your
# provisioned manifests use different names.
TINY_TRAIN_MANIFEST = MANIFEST_ROOT / "tiny_genimage_train.csv"
TINY_DEV_MANIFEST = MANIFEST_ROOT / "tiny_genimage_dev.csv"
GENIMAGE_UNSEEN_MANIFEST = MANIFEST_ROOT / "genimage_unseen_external.csv"
SYNTHBUSTER_MANIFEST = MANIFEST_ROOT / "synthbuster_external.csv"

writable_roots = [CACHE_ROOT, OUTPUT_ROOT, EXTERNAL_ROOT]
if BUILD_MANIFESTS:
    writable_roots.append(MANIFEST_ROOT)
for writable in writable_roots:
    writable.mkdir(parents=True, exist_ok=True)

if not BUILD_MANIFESTS and MANIFEST_ROOT.is_relative_to(KAGGLE_INPUT_ROOT):
    print("manifest root is read-only; manifests must already exist there")

print("data root    : (read-only)", DATA_ROOT)
print("manifest root:", "(writable)" if BUILD_MANIFESTS else "(read-only)", MANIFEST_ROOT)
print("npr ckpt     : (read-only)", NPR_CHECKPOINT_PATH)
print("hf cache     :", HF_CACHE_ROOT if HF_CACHE_ROOT else "(default, ephemeral)")
print("cache root   : (writable)", CACHE_ROOT)
print("output root  : (writable)", OUTPUT_ROOT)
print("external root: (writable)", EXTERNAL_ROOT)
print("build manifests:", BUILD_MANIFESTS)
"""

STORAGE_CODE_KAGGLE_VERTEX = STORAGE_CODE_KAGGLE.replace(
    'INPUT_CHECKPOINT_DIR = KAGGLE_INPUT_ROOT / "<your-npr-checkpoint-dataset>"',
    'INPUT_CHECKPOINT_DIR = KAGGLE_WORKING_ROOT / "npr-checkpoint"',
)

NPR_CHECKPOINT_MD = """
### 5b. NPR checkpoint

The Vertex quick notebook can fetch the official NPR checkpoint into writable
Kaggle storage. Re-running this cell is safe: if `NPR.pth` already exists and is
non-empty, the download is skipped.
"""

NPR_CHECKPOINT_CODE = """
NPR_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/chuangchuangtan/"
    "NPR-DeepfakeDetection/main/NPR.pth"
)

NPR_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
if NPR_CHECKPOINT_PATH.exists() and NPR_CHECKPOINT_PATH.stat().st_size > 0:
    print("NPR checkpoint already exists; skipping download")
else:
    subprocess.run(
        ["wget", "-O", str(NPR_CHECKPOINT_PATH), NPR_CHECKPOINT_URL],
        check=True,
    )

print(
    "NPR checkpoint:",
    NPR_CHECKPOINT_PATH,
    NPR_CHECKPOINT_PATH.exists(),
    "bytes:",
    NPR_CHECKPOINT_PATH.stat().st_size if NPR_CHECKPOINT_PATH.exists() else 0,
)
"""

CONFIG_MD = """
## 6. Generate the runtime config

`configs/phase_ab.yaml` is treated as a **read-only template**. This cell copies
it and rewrites path values only, then writes the result under
`.cache/aiforensics-notebook/` inside the repository.

Two constraints drive that location:

- the config loader finds the repository root by walking up to `pyproject.toml`,
  so the generated file must stay under `REPO_ROOT`;
- `.cache/` is git-ignored, so the generated config is never committed.

Scientific settings are **not** touched: dataset enable flags, model ids, prompt
ids, CLIP seeds, the metric list, report policy, and the pinned NPR repository
URL/commit all stay exactly as committed.
"""

CONFIG_CODE_TEMPLATE = """
import yaml

TEMPLATE_CONFIG = REPO_ROOT / "configs/phase_ab.yaml"
GENERATED_CONFIG = REPO_ROOT / ".cache/aiforensics-notebook/phase_ab_{platform}.yaml"

with open(TEMPLATE_CONFIG, encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

# Only environment/path values change.
cfg["paths"]["data_root"] = str(DATA_ROOT)
cfg["paths"]["manifest_root"] = str(MANIFEST_ROOT)
cfg["paths"]["cache_root"] = str(CACHE_ROOT)
cfg["paths"]["output_root"] = str(OUTPUT_ROOT)
cfg["paths"]["external_root"] = str(EXTERNAL_ROOT)

cfg["datasets"]["tiny_genimage"]["train_manifest"] = str(TINY_TRAIN_MANIFEST)
cfg["datasets"]["tiny_genimage"]["dev_manifest"] = str(TINY_DEV_MANIFEST)
cfg["datasets"]["genimage_unseen"]["manifest"] = str(GENIMAGE_UNSEEN_MANIFEST)
cfg["datasets"]["synthbuster"]["manifest"] = str(SYNTHBUSTER_MANIFEST)

cfg["baselines"]["npr"]["checkpoint_path"] = str(NPR_CHECKPOINT_PATH)

GENERATED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
with open(GENERATED_CONFIG, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

os.environ["AIF_CONFIG"] = str(GENERATED_CONFIG)
print("runtime config:", GENERATED_CONFIG)
"""

VALIDATE_MD = """
## 7. Validate provisioned inputs

This cell only reports where things are. It does not read image data, compute
checksums, or validate manifest contents: `aiforensics prepare` remains the
authoritative validator.

When `BUILD_MANIFESTS` is true the manifest CSVs are expected to be **absent**
here; section 8 creates them. The cell also lists the generator directories it
can see under `DATA_ROOT` and checks them against the generators the config asks
for, because a wrong `DATA_ROOT` is the most common first-run mistake.

A missing NPR checkpoint is surfaced here, before the NPR command runs. Do not
download a checkpoint from an unverified third party.
"""

VALIDATE_CODE = """
enabled_manifests = []
if cfg["datasets"]["tiny_genimage"]["enabled"]:
    enabled_manifests += [TINY_TRAIN_MANIFEST, TINY_DEV_MANIFEST]
if cfg["datasets"]["genimage_unseen"]["enabled"]:
    enabled_manifests.append(GENIMAGE_UNSEEN_MANIFEST)
if cfg["datasets"]["synthbuster"]["enabled"]:
    enabled_manifests.append(SYNTHBUSTER_MANIFEST)

print("runtime config :", GENERATED_CONFIG, "exists:", GENERATED_CONFIG.is_file())
print("data root      :", DATA_ROOT, "exists:", DATA_ROOT.is_dir())
print("manifest root  :", MANIFEST_ROOT, "exists:", MANIFEST_ROOT.is_dir())
print("cache root     :", CACHE_ROOT, "exists:", CACHE_ROOT.is_dir())
print("output root    :", OUTPUT_ROOT, "exists:", OUTPUT_ROOT.is_dir())
print("external root  :", EXTERNAL_ROOT, "exists:", EXTERNAL_ROOT.is_dir())

# Generator directories are the dataset layout contract: a directory holding at
# least one of the dataset-native split directories.
split_dirs = ("train", "val")
found_generators = []
if DATA_ROOT.is_dir():
    for entry in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        if any((entry / split).is_dir() for split in split_dirs):
            found_generators.append(entry.name)

print("\\ngenerator directories under data root:", len(found_generators))
for name in found_generators:
    print("  -", name)
if not found_generators and BUILD_MANIFESTS:
    print(
        "  none found: check DATA_ROOT. Expected "
        "<DATA_ROOT>/<generator>/<train|val>/<ai|nature>/"
    )

configured_generators = []
if cfg["datasets"]["tiny_genimage"]["enabled"]:
    configured_generators += cfg["datasets"]["tiny_genimage"].get("generators", [])
if cfg["datasets"]["genimage_unseen"]["enabled"]:
    configured_generators += cfg["datasets"]["genimage_unseen"].get("generators", [])

missing_generators = [g for g in configured_generators if g not in found_generators]
print("\\nconfigured generators:", configured_generators or "none")
if missing_generators:
    print("  MISSING under data root:", missing_generators)
    print("  prepare --build-manifests will fail until DATA_ROOT or the config matches")

print("\\nmanifests for enabled datasets:")
for path in enabled_manifests:
    print("  -", path, "exists:", path.is_file())
if BUILD_MANIFESTS:
    print("  (BUILD_MANIFESTS is true: section 8 creates/overwrites these)")

print("\\nnpr checkpoint :", NPR_CHECKPOINT_PATH, "exists:", NPR_CHECKPOINT_PATH.is_file())
if not NPR_CHECKPOINT_PATH.is_file():
    print(
        "  NPR will follow its allow_deferred policy: provide the official "
        "checkpoint at this path for a real NPR run."
    )
"""

FULL_RUN_MD = """
## 8. Full Phase A/B run

These cells call the public CLI in the required order. Each cell uses
`set -euo pipefail`, so a real CLI failure stops the cell and stays visible.
Nothing is wrapped in `|| true`.

The first cell runs `prepare`. When `BUILD_MANIFESTS` is true it passes
`--build-manifests`, which reads the GenImage-layout `DATA_ROOT`
(`<generator>/<train|val>/<ai|nature>/`) and **overwrites** the configured
manifest CSVs before validating them. Which generators are in-distribution
versus held out comes from `configs/phase_ab.yaml`, not from this notebook.

`assisted_qwen` must run **after** `clip_probe` because its Phase A/B contract
consumes CLIP assistant predictions.

One `clip_probe` command covers every configured seed; the notebook never loops
over seeds itself. A baseline that records a `deferred` artifact according to its
adapter contract is a legitimate environment outcome, not a notebook error to
swallow.
"""

# Exported for the prepare cell: keeps the build decision in one place instead of
# duplicating the flag inside a shell command.
PREPARE_ENV_CODE = """
# AIF_SECTION: full_run
os.environ["AIF_PREPARE_ARGS"] = "--build-manifests" if BUILD_MANIFESTS else ""
print("prepare args:", os.environ["AIF_PREPARE_ARGS"] or "(validate only)")
"""

FULL_RUN_CELLS = [
    'aiforensics prepare ${AIF_PREPARE_ARGS} --config "$AIF_CONFIG"',
    'aiforensics run --baseline clip_probe --config "$AIF_CONFIG"',
    'aiforensics run --baseline qwen_vl --config "$AIF_CONFIG"',
    'aiforensics run --baseline npr --config "$AIF_CONFIG"',
    'aiforensics run --baseline assisted_qwen --config "$AIF_CONFIG"',
    'aiforensics evaluate --config "$AIF_CONFIG"',
    'aiforensics report --config "$AIF_CONFIG"',
]

VERTEX_CLI_CELLS = [
    'aiforensics prepare ${AIF_PREPARE_ARGS} --config "$AIF_CONFIG"',
    'aiforensics run --baseline clip_probe --config "$AIF_CONFIG"',
    'aiforensics run --baseline qwen_vl --config "$AIF_CONFIG"',
    'aiforensics run --baseline npr --config "$AIF_CONFIG"',
    'aiforensics run --baseline assisted_qwen --config "$AIF_CONFIG"',
    'aiforensics evaluate --config "$AIF_CONFIG"',
    'aiforensics report --config "$AIF_CONFIG"',
]

VERTEX_AUTH_MD = """
## 8. Vertex Qwen endpoint preflight

This validates the cloud Qwen path before the CLI runtime is changed.
Authentication comes from the Kaggle Secret named
`GOOGLE_APPLICATION_CREDENTIALS`. The secret value is the service-account JSON
payload. The notebook parses it in memory, refreshes Google credentials, and
passes the resulting bearer token to the OpenAI-compatible client.

Credential material is never printed.
"""

VERTEX_AUTH_CODE = """
# AIF_SECTION: vertex_qwen
import json

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from kaggle_secrets import UserSecretsClient
from openai import OpenAI

VERTEX_PROJECT_ID = "579187260419"
VERTEX_LOCATION = "asia-southeast1"
VERTEX_ENDPOINT_ID = "mg-endpoint-ccc259a2-c268-4ca6-8713-3bcdaaaf5909"
VERTEX_ENDPOINT_DOMAIN = (
    "mg-endpoint-ccc259a2-c268-4ca6-8713-3bcdaaaf5909."
    "asia-southeast1-635507464424.prediction.vertexai.goog"
)
VERTEX_MODEL_ID = "qwen2_5-vl-7b-instruct-1788570383931"
VERTEX_ENDPOINT_PATH = (
    "/v1/projects/579187260419/locations/asia-southeast1/endpoints/"
    "mg-endpoint-ccc259a2-c268-4ca6-8713-3bcdaaaf5909"
)
EXPECTED_BASE_URL = f"https://{VERTEX_ENDPOINT_DOMAIN}{VERTEX_ENDPOINT_PATH}"


def build_vertex_base_url(project_id: str, location: str, endpoint_id: str, domain: str) -> str:
    cleaned_domain = domain.removeprefix("https://").rstrip("/")
    if cleaned_domain == "aiplatform.googleapis.com":
        raise ValueError("Dedicated Endpoint runs must use the prediction.vertexai.goog domain")
    if not cleaned_domain.endswith(".prediction.vertexai.goog"):
        raise ValueError(f"Unexpected Vertex dedicated endpoint domain: {cleaned_domain}")
    return (
        f"https://{cleaned_domain}/v1/projects/{project_id}"
        f"/locations/{location}/endpoints/{endpoint_id}"
    )


BASE_URL = build_vertex_base_url(
    VERTEX_PROJECT_ID,
    VERTEX_LOCATION,
    VERTEX_ENDPOINT_ID,
    VERTEX_ENDPOINT_DOMAIN,
)
if BASE_URL != EXPECTED_BASE_URL:
    raise RuntimeError(f"Unexpected Vertex base URL: {BASE_URL}")

user_secrets = UserSecretsClient()
service_account_json = user_secrets.get_secret("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["AIF_GOOGLE_APPLICATION_CREDENTIALS_JSON"] = service_account_json
service_account_info = json.loads(service_account_json)
credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
credentials.refresh(Request())

client = OpenAI(
    api_key=credentials.token,
    base_url=BASE_URL,
)

print("vertex base url:", BASE_URL)
print("vertex model id:", VERTEX_MODEL_ID)
print("vertex credentials: loaded and refreshed")
"""

VERTEX_CONFIG_CODE_TEMPLATE = """
import yaml

TEMPLATE_CONFIG = REPO_ROOT / "configs/phase_ab_vertex_quick.yaml"
GENERATED_CONFIG = REPO_ROOT / ".cache/aiforensics-notebook/phase_ab_{platform}.yaml"

with open(TEMPLATE_CONFIG, encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

# Only environment/path values change; quick scientific limits stay in the
# dedicated quick config so this notebook can prove the pipeline cheaply.
cfg["paths"]["data_root"] = str(DATA_ROOT)
cfg["paths"]["manifest_root"] = str(MANIFEST_ROOT)
cfg["paths"]["cache_root"] = str(CACHE_ROOT)
cfg["paths"]["output_root"] = str(OUTPUT_ROOT)
cfg["paths"]["external_root"] = str(EXTERNAL_ROOT)

cfg["datasets"]["tiny_genimage"]["train_manifest"] = str(TINY_TRAIN_MANIFEST)
cfg["datasets"]["tiny_genimage"]["dev_manifest"] = str(TINY_DEV_MANIFEST)
cfg["datasets"]["genimage_unseen"]["manifest"] = str(GENIMAGE_UNSEEN_MANIFEST)
cfg["datasets"]["synthbuster"]["manifest"] = str(SYNTHBUSTER_MANIFEST)

cfg["baselines"]["npr"]["checkpoint_path"] = str(NPR_CHECKPOINT_PATH)

cfg["baselines"]["qwen_vl"]["provider"] = "vertex_openai"
cfg["baselines"]["assisted_qwen"]["provider"] = "vertex_openai"
cfg["report"]["filename"] = "phase_ab_vertex_quick_report.md"

GENERATED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
with open(GENERATED_CONFIG, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

os.environ["AIF_CONFIG"] = str(GENERATED_CONFIG)
print("runtime config:", GENERATED_CONFIG)
print("report filename:", cfg["report"]["filename"])
"""

VERTEX_QWEN_MD = """
## 9. Call Qwen through Vertex

This is a one-image endpoint preflight before the full CLI run. The real
evaluation still happens through `aiforensics run --baseline qwen_vl` and
`aiforensics run --baseline assisted_qwen`, which write `predictions.jsonl` for
`evaluate` and `report`.
"""

VERTEX_QWEN_CODE = """
# AIF_SECTION: vertex_qwen
import base64
import mimetypes

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def first_image_under(root: Path) -> Path:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    raise FileNotFoundError(f"No image found under {root}")


def image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


VERTEX_QWEN_PROMPT = \"\"\"You are an image-forensics classifier.

Classify the provided image as either "real" or "fake".
Return exactly one JSON object with keys: label, confidence, evidence.
label must be "real" or "fake"; confidence must be a number from 0 to 1.
\"\"\"

VERTEX_TEST_IMAGE = first_image_under(DATA_ROOT)
print("vertex test image:", VERTEX_TEST_IMAGE)

response = client.chat.completions.create(
    model="qwen2_5-vl-7b-instruct-1788570383931",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": VERTEX_QWEN_PROMPT},
                {"type": "image_url", "image_url": {"url": image_data_url(VERTEX_TEST_IMAGE)}},
            ],
        }
    ],
    temperature=0,
    max_tokens=256,
)

print(response.choices[0].message.content)
"""

VERTEX_RUN_MD = """
## 10. Quick Phase A/B run with Vertex Qwen

This quick run uses `configs/phase_ab_vertex_quick.yaml`: CLIP trains one seed,
the image caps are small, and both Qwen baselines call the Dedicated Vertex AI
Endpoint. The final report therefore includes Qwen Vertex metrics whenever the
endpoint and credentials are available.
"""

STORAGE_MD_KAGGLE_ALL_VAL = """
## 5. Storage inputs

Kaggle separates **read-only** attached data from **writable** working storage:

- `/kaggle/input/<your-dataset>` is read-only. Research images live here.
- `/kaggle/working` is writable but ephemeral; save notebook output to keep artifacts.

This notebook evaluates Qwen-VL via Vertex AI across all generators in the dataset
`val` split only (`<DATA_ROOT>/<generator>/val/{ai,nature}/*`).
"""

STORAGE_CODE_KAGGLE_ALL_VAL = """
# User-editable: Kaggle mount points.
KAGGLE_INPUT_ROOT = Path("/kaggle/input")  # read-only attached datasets
KAGGLE_WORKING_ROOT = Path("/kaggle/working")  # writable, ephemeral

# Full path to the attached dataset directory that holds the images.
# Expected GenImage layout: <DATA_ROOT>/<generator>/val/{ai,nature}/*
INPUT_DATA_DIR = KAGGLE_INPUT_ROOT / "datasets/yangsangtai/tiny-genimage"

# Maximum number of images per generator to evaluate.
# 0 means full val (all available val images for each generator).
# Set to a small integer (e.g. 10 or 50) for fast smoke/sanity checks.
MAX_IMAGES_PER_GENERATOR = 0

# Set True to build manifests from the val split of all generators in INPUT_DATA_DIR;
# set False when manifest CSV is already provisioned.
BUILD_MANIFESTS = True
INPUT_MANIFEST_DIR = KAGGLE_INPUT_ROOT / "<your-manifests-dataset>"

# Read-only inputs.
DATA_ROOT = INPUT_DATA_DIR

# Writable outputs: never place these under /kaggle/input.
CACHE_ROOT = KAGGLE_WORKING_ROOT / "cache"
OUTPUT_ROOT = Path("/kaggle/working/outputs-qwen-all-val")
EXTERNAL_ROOT = KAGGLE_WORKING_ROOT / "external"

# Built manifests must land in writable storage; provisioned ones stay read-only.
MANIFEST_ROOT = KAGGLE_WORKING_ROOT / "manifests" if BUILD_MANIFESTS else INPUT_MANIFEST_DIR

# Dedicated manifest for all-val evaluation
GENIMAGE_ALL_VAL_MANIFEST = MANIFEST_ROOT / "genimage_all_val.csv"

writable_roots = [CACHE_ROOT, OUTPUT_ROOT, EXTERNAL_ROOT]
if BUILD_MANIFESTS:
    writable_roots.append(MANIFEST_ROOT)
for writable in writable_roots:
    writable.mkdir(parents=True, exist_ok=True)

if not BUILD_MANIFESTS and MANIFEST_ROOT.is_relative_to(KAGGLE_INPUT_ROOT):
    print("manifest root is read-only; manifests must already exist there")

print("data root    : (read-only)", DATA_ROOT)
print("manifest root:", "(writable)" if BUILD_MANIFESTS else "(read-only)", MANIFEST_ROOT)
print("cache root   : (writable)", CACHE_ROOT)
print("output root  : (writable)", OUTPUT_ROOT)
print("external root: (writable)", EXTERNAL_ROOT)
print("max images/generator:", MAX_IMAGES_PER_GENERATOR, "(0 = full val)")
print("build manifests:", BUILD_MANIFESTS)
"""

CONFIG_MD_ALL_VAL = """
## 6. Generate the runtime config

`configs/qwen_vertex_all_val.yaml` is treated as a **read-only template**. This cell copies
it, discovers all generators with a `val` split under `DATA_ROOT`, and rewrites path and
generator values under `.cache/aiforensics-notebook/` inside the repository.
"""

VERTEX_CONFIG_ALL_VAL_CODE_TEMPLATE = """
import yaml

TEMPLATE_CONFIG = REPO_ROOT / "configs/qwen_vertex_all_val.yaml"
GENERATED_CONFIG = REPO_ROOT / ".cache/aiforensics-notebook/qwen_vertex_all_val_{platform}.yaml"

with open(TEMPLATE_CONFIG, encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

# Relocate paths to runtime storage
cfg["paths"]["data_root"] = str(DATA_ROOT)
cfg["paths"]["manifest_root"] = str(MANIFEST_ROOT)
cfg["paths"]["cache_root"] = str(CACHE_ROOT)
cfg["paths"]["output_root"] = str(OUTPUT_ROOT)
cfg["paths"]["external_root"] = str(EXTERNAL_ROOT)


def discover_generator_dirs(data_root: Path) -> list[str]:
    \"\"\"Discover all generator directories that have a val split.\"\"\"
    if not data_root.is_dir():
        return []
    found = []
    for entry in sorted(data_root.iterdir()):
        if entry.is_dir() and (entry / "val").is_dir():
            found.append(entry.name)
    return found


all_val_generators = discover_generator_dirs(DATA_ROOT)

cfg["datasets"]["genimage_unseen"]["generators"] = all_val_generators
cfg["datasets"]["genimage_unseen"]["manifest"] = str(GENIMAGE_ALL_VAL_MANIFEST)
cfg["datasets"]["genimage_unseen"]["source_split"] = "val"
cfg["datasets"]["genimage_unseen"]["max_images"] = int(MAX_IMAGES_PER_GENERATOR)

cfg["baselines"]["qwen_vl"]["provider"] = "vertex_openai"
cfg["report"]["filename"] = "qwen_vertex_all_val_report.md"

GENERATED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
with open(GENERATED_CONFIG, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

os.environ["AIF_CONFIG"] = str(GENERATED_CONFIG)
print("runtime config:", GENERATED_CONFIG)
print("report filename:", cfg["report"]["filename"])
print("generators (val only):", all_val_generators)
print("max images per generator:", MAX_IMAGES_PER_GENERATOR)
"""

VALIDATE_MD_ALL_VAL = """
## 7. Validate provisioned inputs

This cell checks the runtime paths and lists all discovered generator directories
with a `val` split.
"""

VALIDATE_CODE_ALL_VAL = """
print("runtime config :", GENERATED_CONFIG, "exists:", GENERATED_CONFIG.is_file())
print("data root      :", DATA_ROOT, "exists:", DATA_ROOT.is_dir())
print("manifest root  :", MANIFEST_ROOT, "exists:", MANIFEST_ROOT.is_dir())
print("cache root     :", CACHE_ROOT, "exists:", CACHE_ROOT.is_dir())
print("output root    :", OUTPUT_ROOT, "exists:", OUTPUT_ROOT.is_dir())

found_val_generators = []
if DATA_ROOT.is_dir():
    for entry in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        if (entry / "val").is_dir():
            found_val_generators.append(entry.name)

print("\\ngenerator directories with val/ under data root:", len(found_val_generators))
for name in found_val_generators:
    print("  -", name)
if not found_val_generators and BUILD_MANIFESTS:
    print(
        "  none found: check DATA_ROOT. Expected "
        "<DATA_ROOT>/<generator>/val/<ai|nature>/"
    )

print(
    "\\nall-val manifest target:",
    GENIMAGE_ALL_VAL_MANIFEST,
    "exists:",
    GENIMAGE_ALL_VAL_MANIFEST.is_file(),
)
if BUILD_MANIFESTS:
    print("  (BUILD_MANIFESTS is true: prepare --build-manifests will create/overwrite this)")
"""

VERTEX_ALL_VAL_RUN_MD = """
## 10. Run Qwen-VL across all generators (val split)

These cells execute the pipeline for `qwen_vl` only. Neither CLIP, NPR, nor Assisted Qwen
are executed.

1. `prepare` reads the GenImage layout under `DATA_ROOT`, filters solely the `val` split,
   and writes `genimage_all_val.csv`.
2. `run --baseline qwen_vl` classifies each image in parallel through the Vertex AI Dedicated
   Endpoint.
3. `evaluate` computes classification metrics (accuracy, AUROC, log-loss, etc.) by generator.
4. `report` renders `qwen_vertex_all_val_report.md`.
"""

VERTEX_ALL_VAL_CLI_CELLS = [
    'aiforensics prepare ${AIF_PREPARE_ARGS} --config "$AIF_CONFIG"',
    'aiforensics run --baseline qwen_vl --config "$AIF_CONFIG"',
    'aiforensics evaluate --config "$AIF_CONFIG"',
    'aiforensics report --config "$AIF_CONFIG"',
]


ARTIFACT_MD = """
## 9. Artifacts

Where to look after a run. The notebook only points at these files; parsing and
rendering stay in the package (`aiforensics evaluate` and `aiforensics report`).

```text
<OUTPUT_ROOT>/manifest_validation.json
<OUTPUT_ROOT>/<run_id>/status.json
<OUTPUT_ROOT>/<run_id>/predictions.jsonl
<OUTPUT_ROOT>/<run_id>/metrics.json
<OUTPUT_ROOT>/<run_id>/metrics_by_source.csv
<OUTPUT_ROOT>/<configured report filename>
```
"""

ARTIFACT_CODE = """
report_path = OUTPUT_ROOT / cfg["report"]["filename"]

print("manifest validation:", OUTPUT_ROOT / "manifest_validation.json")
print("report             :", report_path, "exists:", report_path.is_file())

print("\\nrun directories under", OUTPUT_ROOT)
if OUTPUT_ROOT.is_dir():
    for entry in sorted(p for p in OUTPUT_ROOT.iterdir() if p.is_dir()):
        print("  -", entry.name)
"""

SMOKE_MD = """
## 10. Optional: smoke verification

The smoke flow proves that installation and the CLI pipeline work in this hosted
environment. It uses the committed `configs/phase_ab_smoke.yaml` fixtures.

> Smoke metrics are pipeline checks, **not scientific evidence**.

The committed smoke config is never modified. This section generates a copy that
relocates `cache_root` and `output_root` only, so nothing is written to
repository paths that may be read-only or ephemeral. Smoke `data_root` and smoke
manifests keep pointing at the repository fixtures, because those fixtures *are*
the smoke dataset.
"""

SMOKE_CONFIG_CODE_TEMPLATE = """
# AIF_SECTION: smoke
SMOKE_TEMPLATE = REPO_ROOT / "configs/phase_ab_smoke.yaml"
SMOKE_CONFIG = REPO_ROOT / ".cache/aiforensics-notebook/phase_ab_smoke_{platform}.yaml"

with open(SMOKE_TEMPLATE, encoding="utf-8") as handle:
    smoke_cfg = yaml.safe_load(handle)

# Relocate writable roots only; fixtures stay where they are committed.
smoke_cfg["paths"]["cache_root"] = str(CACHE_ROOT / "smoke")
smoke_cfg["paths"]["output_root"] = str(OUTPUT_ROOT / "smoke")

SMOKE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
with open(SMOKE_CONFIG, "w", encoding="utf-8") as handle:
    yaml.safe_dump(smoke_cfg, handle, sort_keys=False)

os.environ["AIF_SMOKE_CONFIG"] = str(SMOKE_CONFIG)
print("smoke runtime config:", SMOKE_CONFIG)
"""

SMOKE_RUN_CODE = """
%%bash
# AIF_SECTION: smoke
set -euo pipefail

cd "$AIF_REPO_ROOT"
aiforensics prepare --config "$AIF_SMOKE_CONFIG"
aiforensics run --baseline clip_probe --config "$AIF_SMOKE_CONFIG"
aiforensics run --baseline qwen_vl --config "$AIF_SMOKE_CONFIG"
aiforensics run --baseline npr --config "$AIF_SMOKE_CONFIG"
aiforensics run --baseline assisted_qwen --config "$AIF_SMOKE_CONFIG"
aiforensics evaluate --config "$AIF_SMOKE_CONFIG"
aiforensics report --config "$AIF_SMOKE_CONFIG"
"""


def full_run_cell(command: str) -> dict:
    return code(
        f'%%bash\n# AIF_SECTION: full_run\nset -euo pipefail\n\ncd "$AIF_REPO_ROOT"\n{command}\n'
    )


def build(platform: str) -> dict:
    is_colab = platform == "colab"
    cells: list[dict] = [
        md(TITLE_COLAB if is_colab else TITLE_KAGGLE),
        md(PREFLIGHT_MD),
        code(
            PREFLIGHT_CODE
            if is_colab
            else PREFLIGHT_CODE.replace(
                'CLI_VENV_PATH = Path("/content/aiforensics-venv310")',
                'CLI_VENV_PATH = Path("/kaggle/working/aiforensics-venv310")',
            )
        ),
        md(PROVISION_MD),
        code(PROVISION_CODE),
        md(VERIFY_PY_MD),
        code(VERIFY_PY_CODE),
        md(REPO_MD_COLAB if is_colab else REPO_MD_KAGGLE),
        code(
            REPO_CODE_TEMPLATE.format(
                repo_root="/content/ai-image-forensics"
                if is_colab
                else "/kaggle/working/ai-image-forensics"
            )
        ),
        md(INSTALL_MD),
        code(INSTALL_ENV_CODE),
        code(INSTALL_CODE),
        md(VERIFY_CLI_MD),
        code(VERIFY_CLI_CODE),
        md(STORAGE_MD_COLAB if is_colab else STORAGE_MD_KAGGLE),
        code(STORAGE_CODE_COLAB if is_colab else STORAGE_CODE_KAGGLE),
        md(CONFIG_MD),
        code(CONFIG_CODE_TEMPLATE.format(platform=platform)),
        md(VALIDATE_MD),
        code(VALIDATE_CODE),
        md(FULL_RUN_MD),
        code(PREPARE_ENV_CODE),
        *[full_run_cell(command) for command in FULL_RUN_CELLS],
        md(ARTIFACT_MD),
        code(ARTIFACT_CODE),
        md(SMOKE_MD),
        code(SMOKE_CONFIG_CODE_TEMPLATE.format(platform=platform)),
        code(SMOKE_RUN_CODE),
    ]
    return notebook(cells)


def build_kaggle_vertex_qwen() -> dict:
    cells: list[dict] = [
        md(TITLE_KAGGLE_VERTEX_QWEN),
        md(PREFLIGHT_MD),
        code(
            PREFLIGHT_CODE.replace(
                'CLI_VENV_PATH = Path("/content/aiforensics-venv310")',
                'CLI_VENV_PATH = Path("/kaggle/working/aiforensics-venv310")',
            )
        ),
        md(PROVISION_MD),
        code(PROVISION_CODE),
        md(VERIFY_PY_MD),
        code(VERIFY_PY_CODE),
        md(REPO_MD_KAGGLE),
        code(REPO_CODE_TEMPLATE.format(repo_root="/kaggle/working/ai-image-forensics")),
        md(INSTALL_MD),
        code(INSTALL_ENV_CODE),
        code(INSTALL_VERTEX_CODE),
        md(VERIFY_CLI_MD),
        code(VERIFY_CLI_CODE),
        md(STORAGE_MD_KAGGLE),
        code(STORAGE_CODE_KAGGLE_VERTEX),
        md(NPR_CHECKPOINT_MD),
        code(NPR_CHECKPOINT_CODE),
        md(CONFIG_MD),
        code(VERTEX_CONFIG_CODE_TEMPLATE.format(platform="kaggle_vertex_qwen")),
        md(VALIDATE_MD),
        code(VALIDATE_CODE),
        md(VERTEX_AUTH_MD),
        code(VERTEX_AUTH_CODE),
        md(VERTEX_QWEN_MD),
        code(VERTEX_QWEN_CODE),
        md(VERTEX_RUN_MD),
        code(PREPARE_ENV_CODE),
        *[full_run_cell(command) for command in VERTEX_CLI_CELLS],
        md(ARTIFACT_MD),
        code(ARTIFACT_CODE),
    ]
    return notebook(cells)


def build_kaggle_vertex_qwen_all_val() -> dict:
    cells: list[dict] = [
        md(TITLE_KAGGLE_VERTEX_QWEN_ALL_VAL),
        md(PREFLIGHT_MD),
        code(
            PREFLIGHT_CODE.replace(
                'CLI_VENV_PATH = Path("/content/aiforensics-venv310")',
                'CLI_VENV_PATH = Path("/kaggle/working/aiforensics-venv310")',
            )
        ),
        md(PROVISION_MD),
        code(PROVISION_CODE),
        md(VERIFY_PY_MD),
        code(VERIFY_PY_CODE),
        md(REPO_MD_KAGGLE),
        code(REPO_CODE_TEMPLATE.format(repo_root="/kaggle/working/ai-image-forensics")),
        md(INSTALL_MD),
        code(INSTALL_ENV_CODE),
        code(INSTALL_VERTEX_ALL_VAL_CODE),
        md(VERIFY_CLI_MD),
        code(VERIFY_CLI_CODE),
        md(STORAGE_MD_KAGGLE_ALL_VAL),
        code(STORAGE_CODE_KAGGLE_ALL_VAL),
        md(CONFIG_MD_ALL_VAL),
        code(VERTEX_CONFIG_ALL_VAL_CODE_TEMPLATE.format(platform="kaggle")),
        md(VALIDATE_MD_ALL_VAL),
        code(VALIDATE_CODE_ALL_VAL),
        md(VERTEX_AUTH_MD),
        code(VERTEX_AUTH_CODE),
        md(VERTEX_QWEN_MD),
        code(VERTEX_QWEN_CODE),
        md(VERTEX_ALL_VAL_RUN_MD),
        code(PREPARE_ENV_CODE),
        *[full_run_cell(command) for command in VERTEX_ALL_VAL_CLI_CELLS],
        md(ARTIFACT_MD),
        code(ARTIFACT_CODE),
    ]
    return notebook(cells)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for platform in ("colab", "kaggle"):
        path = NOTEBOOK_DIR / f"{platform}_phase_ab.ipynb"
        path.write_text(json.dumps(build(platform), indent=1) + "\n", encoding="utf-8")
        print("wrote", path)
    vertex_path = NOTEBOOK_DIR / "kaggle_vertex_qwen_phase_ab.ipynb"
    vertex_path.write_text(
        json.dumps(build_kaggle_vertex_qwen(), indent=1) + "\n", encoding="utf-8"
    )
    print("wrote", vertex_path)
    vertex_all_val_path = NOTEBOOK_DIR / "kaggle_vertex_qwen_all_val.ipynb"
    vertex_all_val_path.write_text(
        json.dumps(build_kaggle_vertex_qwen_all_val(), indent=1) + "\n", encoding="utf-8"
    )
    print("wrote", vertex_all_val_path)


if __name__ == "__main__":
    main()
