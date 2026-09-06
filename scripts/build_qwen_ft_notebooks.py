"""One-shot generator for the qwen_ft export / submit / eval thin notebooks.

Run from the repository root:

    uv run python scripts/build_qwen_ft_notebooks.py

Notebook JSON is generated rather than hand-edited so that committed cells stay
free of execution counts, outputs, and stale metadata. The notebooks are thin
wrappers: they call ``scripts/export_genimage_qwen_ft_subset.py``,
``scripts/submit_qwen_ft_vertex_job.py``, and the public ``aiforensics`` CLI;
no business logic lives in the cells.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

# Reuse the cell helpers from the existing notebook builder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import code, md, notebook  # noqa: E402

BUCKET_URI = "gs://aiforensics-qwen-ft-579187260419"
PROJECT_ID = "579187260419"
LOCATION = "asia-southeast1"

# ---------------------------------------------------------------------------
# shared cells
# ---------------------------------------------------------------------------

TITLE_EXPORT = f"""
# qwen_ft GenImage Export on Kaggle

This notebook is a **thin wrapper** for step 1-3 of the qwen_ft fine-tuning
flow (`docs/runbook-qwen-finetune-vertex.md`): it verifies the GenImage data
root, runs the balanced subset export script, and uploads the selected images,
manifests, and the label-only training JSONL to the project GCS bucket.

It does **not** implement subset selection, checksum computation, GCS layout,
or training. Those live in the package and the Vertex job.

- Bucket: `{BUCKET_URI}`
- Protocol: `protocol-a-small`, 100 train and 50 eval images per label per
  generator.
- Enable **Internet** in the Kaggle notebook settings; attach the GenImage
  dataset (layout `<generator>/<split>/{{ai,nature}}/*`).
"""

TITLE_SUBMIT = f"""
# qwen_ft Vertex Training Submit and Eval

This notebook is a **thin wrapper** for steps 4-11 of
`docs/runbook-qwen-finetune-vertex.md`: it dry-runs and submits the Vertex AI
CustomJob that LoRA fine-tunes Qwen2.5-VL-7B, then runs the `qwen_ft`
evaluation through the public `aiforensics` CLI (eval-small first, eval-full
only after the parse-failure gate passes, then the report).

It does **not** train anything locally; training runs as a Vertex CustomJob on
an A100. Evaluation loads the trained adapter from
`{BUCKET_URI}/checkpoints/<protocol>/final_adapter`.

- Project: `{PROJECT_ID}`, location: `{LOCATION}`
- Enable **Internet**; provide GCP credentials via Application Default
  Credentials or the Kaggle Secret `GOOGLE_APPLICATION_CREDENTIALS`.
"""

INSTALL_MD = """
## 1. Install the repository and dependencies

The notebook clones this repository into writable storage and installs the
package. Nothing is pinned here; dependency names come from `pyproject.toml`.
"""

INSTALL_CODE = """
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/kaggle/working/ai-image-forensics")
REPO_GIT_URL = "https://github.com/Nnguyen-dev2805/ai-image-forensics.git"

if not REPO_ROOT.exists():
    REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO_GIT_URL, str(REPO_ROOT)], check=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "-e", "."],
    cwd=str(REPO_ROOT),
    check=True,
)
print("repository:", REPO_ROOT)
"""

KAGGLE_AUTH_MD = """
## 2. Authenticate GCP (Kaggle)

Reads the service-account JSON from the Kaggle Secret
`GOOGLE_APPLICATION_CREDENTIALS`, writes it to a private key file, and
activates it for both `gcloud storage cp` and the Google Cloud Storage client.
**Credential material is never printed.**
"""

KAGGLE_AUTH_CODE = """
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from kaggle_secrets import UserSecretsClient

service_account_json = UserSecretsClient().get_secret("GOOGLE_APPLICATION_CREDENTIALS")
service_account_info = json.loads(service_account_json)  # validates JSON before use

KEY_FILE = Path(tempfile.mkstemp(prefix="sa-", suffix=".json")[1])
KEY_FILE.write_text(service_account_json, encoding="utf-8")
os.chmod(KEY_FILE, 0o600)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_FILE)

activate_cmd = f"gcloud auth activate-service-account --key-file={KEY_FILE}"
subprocess.run(shlex.split(activate_cmd), check=True)
print("gcloud authenticated as:", service_account_info.get("client_email", "<unknown>"))
"""

EXPORT_VERIFY_MD = """
## 3. Verify the GenImage data root

The export expects the Kaggle dataset layout
`<DATA_ROOT>/<generator>/<train|val>/{ai,nature}/*`. The cell lists the
generator directories it can see; the export script fails loudly when a
configured generator is missing or has too few images.
"""

EXPORT_VERIFY_CODE = """
DATA_ROOT = Path("/kaggle/input/datasets/yangsangtai/tiny-genimage")

if not DATA_ROOT.is_dir():
    raise FileNotFoundError(
        f"DATA_ROOT does not exist: {DATA_ROOT}. Attach the GenImage dataset "
        "and point DATA_ROOT at its directory."
    )

split_dirs = ("train", "val")
found = [
    entry.name
    for entry in sorted(DATA_ROOT.iterdir())
    if entry.is_dir() and any((entry / split).is_dir() for split in split_dirs)
]
print("generator directories:", len(found))
for name in found:
    print("  -", name)
if not found:
    raise FileNotFoundError(
        f"No <generator>/<split> layout under {DATA_ROOT}; check the attachment."
    )
"""

EXPORT_RUN_MD = f"""
## 4. Run the export script

Dry-run first (manifests only), then upload with `--upload`. The script writes
`train.csv`, `eval_small.csv`, and `qwen_train.jsonl` into `WORK_DIR`, copies
the selected images to `{BUCKET_URI}/data/protocol-a-small/`, and uploads the
manifests plus training JSONL alongside them.
"""

EXPORT_RUN_CODE = f"""
import shlex
import subprocess
import sys

EXPORT_CMD = (
    f"{{sys.executable}} scripts/export_genimage_qwen_ft_subset.py "
    f"--data-root {{DATA_ROOT}} "
    f"--bucket-uri {BUCKET_URI} "
    "--protocol protocol-a-small "
    "--work-dir /kaggle/working/qwen-ft-export "
    "--seed 70 "
    "--train-per-label 100 "
    "--eval-per-label 50"
)

print("[dry-run]", EXPORT_CMD)
subprocess.run(shlex.split(EXPORT_CMD), cwd=str(REPO_ROOT), check=True)
"""

EXPORT_UPLOAD_MD = """
## 5. Upload to GCS

Re-runs are idempotent per file (`gcloud storage cp` overwrites), but the
selection is deterministic only for the same seed and data root. Do not change
the seed between a partial and a full upload.
"""

EXPORT_UPLOAD_CODE = f"""
import shlex
import subprocess

subprocess.run(shlex.split(EXPORT_CMD + " --upload"), cwd=str(REPO_ROOT), check=True)

gcs_base = "{BUCKET_URI}/data/protocol-a-small"
print("uploaded GCS paths:")
for name in ("train.csv", "eval_small.csv"):
    print(f"  {{gcs_base}}/manifests/{{name}}")
print(f"  {{gcs_base}}/qwen_train.jsonl")
"""

VERTEX_AUTH_MD = """
## 2. Verify GCP authentication

The submit script uses Application Default Credentials. If the Kaggle Secret
`GOOGLE_APPLICATION_CREDENTIALS` exists, it is activated first (credential
material is never printed); otherwise the cell verifies ADC is already usable.
"""

VERTEX_AUTH_CODE = """
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

try:
    from kaggle_secrets import UserSecretsClient

    service_account_json = UserSecretsClient().get_secret("GOOGLE_APPLICATION_CREDENTIALS")
    service_account_info = json.loads(service_account_json)
    key_file = Path(tempfile.mkstemp(prefix="sa-", suffix=".json")[1])
    key_file.write_text(service_account_json, encoding="utf-8")
    os.chmod(key_file, 0o600)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_file)
    activate_cmd = f"gcloud auth activate-service-account --key-file={key_file}"
    subprocess.run(shlex.split(activate_cmd), check=True)
    print("gcloud authenticated as:", service_account_info.get("client_email", "<unknown>"))
except Exception:
    print("no Kaggle secret found; relying on existing Application Default Credentials")

probe = subprocess.run(
    ["gcloud", "auth", "list", "--format=value(account)"],
    capture_output=True,
    text=True,
    check=True,
)
print("active accounts:", probe.stdout.strip() or "<none>")
"""

DRY_RUN_MD = """
## 3. Dry-run the Vertex CustomJob payload

Prints the exact REST payload that would be submitted: A100 40GB worker,
GCS checkpoint prefix, and the LoRA training flags. Inspect it before
submitting; training costs real money.
"""

DRY_RUN_CODE = f"""
import shlex
import subprocess

SUBMIT_CMD = (
    f"{{sys.executable}} scripts/submit_qwen_ft_vertex_job.py "
    f"--project-id {PROJECT_ID} "
    f"--location {LOCATION} "
    f"--bucket-uri {BUCKET_URI} "
    "--protocol protocol-a-small "
    "--machine-type a2-highgpu-1g "
    "--accelerator-type NVIDIA_TESLA_A100 "
    "--accelerator-count 1"
)

subprocess.run(shlex.split(SUBMIT_CMD + " --dry-run"), cwd=str(REPO_ROOT), check=True)
"""

SUBMIT_MD = """
## 4. Submit the Vertex CustomJob

Submits the job and prints the resource name. Monitor progress with:

```bash
gcloud ai custom-jobs describe <JOB_NAME> --region=asia-southeast1
gcloud ai custom-jobs tail-logs <JOB_NAME> --region=asia-southeast1
```

Checkpoints land under
`{BUCKET_URI}/checkpoints/protocol-a-small/`; the eval cells below require
`final_adapter/adapter_config.json` to exist there.
"""

SUBMIT_CODE = """
import shlex
import subprocess

result = subprocess.run(
    shlex.split(SUBMIT_CMD),
    cwd=str(REPO_ROOT),
    check=True,
    capture_output=True,
    text=True,
)
print(result.stdout)
print(
    "Training runs remotely. Re-run this notebook from section 5 once "
    "final_adapter/adapter_config.json exists in GCS."
)
"""

CONFIG_MD = """
## 5. Generate the runtime eval configs

The committed `configs/qwen_ft_protocol_a_small.yaml` is a read-only template.
This cell writes eval-small and eval-full copies under
`.cache/aiforensics-notebook/` that only relocate the evaluation manifest and
the writable roots. The eval-full manifest is the operator-provided
7000-image all-val CSV produced by the earlier Vertex all-val flow.
"""

CONFIG_CODE = """
import yaml
from pathlib import Path

TEMPLATE = REPO_ROOT / "configs/qwen_ft_protocol_a_small.yaml"
CACHE_NOTEBOOK = REPO_ROOT / ".cache/aiforensics-notebook"

# Operator inputs: writable roots and the full-validation manifest.
OUTPUT_ROOT = Path("/kaggle/working/outputs-qwen-ft")
CACHE_ROOT = Path("/kaggle/working/cache-qwen-ft")
EVAL_FULL_MANIFEST = Path("/kaggle/working/manifests/genimage_all_val.csv")

with open(TEMPLATE, encoding="utf-8") as handle:
    base_cfg = yaml.safe_load(handle)


def write_eval_config(suffix: str, manifest: Path, report_name: str) -> Path:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg["paths"]["manifest_root"] = str(manifest.parent)
    cfg["paths"]["cache_root"] = str(CACHE_ROOT / suffix)
    cfg["paths"]["output_root"] = str(OUTPUT_ROOT / suffix)
    cfg["datasets"]["tiny_genimage"]["train_manifest"] = str(manifest.parent / "train.csv")
    cfg["datasets"]["tiny_genimage"]["dev_manifest"] = str(manifest)
    cfg["datasets"]["genimage_unseen"]["manifest"] = str(manifest)
    cfg["report"]["filename"] = report_name
    out = CACHE_NOTEBOOK / f"qwen_ft_protocol_a_{suffix}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return out


EVAL_SMALL_CONFIG = write_eval_config(
    "eval-small", Path("/kaggle/working/manifests/eval_small.csv"), "qwen_ft_eval_small_report.md"
)
EVAL_FULL_CONFIG = write_eval_config(
    "eval-full", EVAL_FULL_MANIFEST, "qwen_ft_eval_full_report.md"
)

for path in (EVAL_SMALL_CONFIG, EVAL_FULL_CONFIG):
    print("runtime config:", path)
"""

EVAL_SMALL_MD = """
## 6. Run qwen_ft eval-small

Loads the base model plus the trained LoRA adapter and emits label-only
predictions for the 700-image eval-small manifest through the public CLI.
"""

EVAL_SMALL_CODE = """
import shlex
import subprocess

subprocess.run(
    shlex.split(f"aiforensics run --baseline qwen_ft --config {EVAL_SMALL_CONFIG}"),
    cwd=str(REPO_ROOT),
    check=True,
)
subprocess.run(
    shlex.split(f"aiforensics evaluate --config {EVAL_SMALL_CONFIG}"),
    cwd=str(REPO_ROOT),
    check=True,
)
"""

PARSE_GATE_MD = """
## 7. Parse-failure gate

Eval-full costs 7000 inferences; it only runs when the eval-small parse
failure rate is below the 2% success criterion from the design spec.
"""

PARSE_GATE_CODE = """
import json
from pathlib import Path


def latest_qwen_ft_predictions(output_root: Path) -> Path:
    runs = sorted(
        (path for path in output_root.iterdir() if path.is_dir() and "_qwen_ft" in path.name),
        key=lambda path: path.name,
    )
    if not runs:
        raise FileNotFoundError(f"No qwen_ft run directory under {output_root}")
    return runs[-1] / "predictions.jsonl"


pred_path = latest_qwen_ft_predictions(OUTPUT_ROOT / "eval-small")
records = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines()]
total = len(records)
failed = sum(1 for record in records if record.get("parse_status") == "failed")
parse_failure_rate = failed / total if total else 1.0

print(f"parse failure rate: {failed}/{total} = {parse_failure_rate:.4f}")
if parse_failure_rate >= 0.02:
    raise RuntimeError(
        "Parse failure rate is at or above the 2% success criterion; fix the "
        "adapter/parser before spending eval-full inference budget."
    )
print("GATE OK: eval-full may run.")
"""

EVAL_FULL_MD = """
## 8. Run qwen_ft eval-full and generate reports

Runs the full 7000-image validation manifest, evaluates both manifests, and
writes the comparison reports next to the run artifacts.
"""

EVAL_FULL_CODE = """
import shlex
import subprocess

subprocess.run(
    shlex.split(f"aiforensics run --baseline qwen_ft --config {EVAL_FULL_CONFIG}"),
    cwd=str(REPO_ROOT),
    check=True,
)
for config in (EVAL_SMALL_CONFIG, EVAL_FULL_CONFIG):
    subprocess.run(
        shlex.split(f"aiforensics report --config {config}"),
        cwd=str(REPO_ROOT),
        check=True,
    )
"""

ARTIFACT_MD = """
## 9. Artifacts

```text
gs://aiforensics-qwen-ft-579187260419/data/protocol-a-small/...   exported data
gs://aiforensics-qwen-ft-579187260419/checkpoints/protocol-a-small/...  checkpoints
<OUTPUT_ROOT>/<suffix>/<run_id>/predictions.jsonl
<OUTPUT_ROOT>/<suffix>/<run_id>/metrics.json
<OUTPUT_ROOT>/<suffix>/<run_id>/confusion_matrix.json
<OUTPUT_ROOT>/<suffix>/<configured report filename>
```

Keep every artifact: a run without saved artifacts is not a valid experiment
result per the design spec.
"""


def build_export_notebook() -> dict:
    return notebook(
        [
            md(TITLE_EXPORT),
            md(INSTALL_MD),
            code(INSTALL_CODE),
            md(KAGGLE_AUTH_MD),
            code(KAGGLE_AUTH_CODE),
            md(EXPORT_VERIFY_MD),
            code(EXPORT_VERIFY_CODE),
            md(EXPORT_RUN_MD),
            code(EXPORT_RUN_CODE),
            md(EXPORT_UPLOAD_MD),
            code(EXPORT_UPLOAD_CODE),
        ]
    )


def build_submit_notebook() -> dict:
    return notebook(
        [
            md(TITLE_SUBMIT),
            md(INSTALL_MD),
            code(INSTALL_CODE),
            md(VERTEX_AUTH_MD),
            code(VERTEX_AUTH_CODE),
            md(DRY_RUN_MD),
            code(DRY_RUN_CODE),
            md(SUBMIT_MD),
            code(SUBMIT_CODE),
            md(CONFIG_MD),
            code(CONFIG_CODE),
            md(EVAL_SMALL_MD),
            code(EVAL_SMALL_CODE),
            md(PARSE_GATE_MD),
            code(PARSE_GATE_CODE),
            md(EVAL_FULL_MD),
            code(EVAL_FULL_CODE),
            md(ARTIFACT_MD),
        ]
    )


# ---------------------------------------------------------------------------
# Modal zip export notebook (no GCS, no Vertex)
# ---------------------------------------------------------------------------

TITLE_MODAL_EXPORT = """
# qwen_ft GenImage Zip Export on Kaggle (Modal path)

This notebook is a **thin wrapper** for the Modal fine-tuning flow
(`docs/runbook-qwen-finetune-modal.md`): it verifies the GenImage data root,
exports a balanced protocol-relative zip archive, and prints the archive to
download. No GCS bucket, no Vertex endpoint, and no GCP service-account
secret are required.

- Protocol: `protocol-a-small`, 100 train and 50 eval images per label per
  generator, seed 70.
- Output: `/kaggle/working/qwen_ft_protocol_a_small.zip` — save the notebook
  output and download this file, then upload it to Modal Volume
  `aiforensics-qwen-ft` at `/archives/`.
- Enable **Internet** in the Kaggle notebook settings.
"""

MODAL_EXPORT_CODE = """
# AIF_SECTION: modal_zip_export
import shlex
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path("/kaggle/input/datasets/yangsangtai/tiny-genimage")
ARCHIVE_PATH = Path("/kaggle/working/qwen_ft_protocol_a_small.zip")

EXPORT_CMD = (
    f"{sys.executable} scripts/export_genimage_qwen_ft_subset.py "
    f"--data-root {DATA_ROOT} "
    "--protocol protocol-a-small "
    "--train-per-label 100 "
    "--eval-per-label 50 "
    "--seed 70 "
    "--path-mode modal-volume "
    f"--archive-path {ARCHIVE_PATH}"
)

subprocess.run(shlex.split(EXPORT_CMD), cwd=str(REPO_ROOT), check=True)

import zipfile

with zipfile.ZipFile(ARCHIVE_PATH) as zf:
    names = zf.namelist()
print("archive:", ARCHIVE_PATH, ARCHIVE_PATH.stat().st_size, "bytes")
print("zip entries:", len(names))
print("next step: download this file and run")
print("  python3 -m modal volume put aiforensics-qwen-ft "
      f"{ARCHIVE_PATH.name} /archives/{ARCHIVE_PATH.name}")
"""


def build_modal_export_notebook() -> dict:
    return notebook(
        [
            md(TITLE_MODAL_EXPORT),
            md(INSTALL_MD),
            code(INSTALL_CODE),
            md(EXPORT_VERIFY_MD),
            code(EXPORT_VERIFY_CODE),
            md("## 3. Export the protocol zip archive"),
            code(MODAL_EXPORT_CODE),
        ]
    )


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in (
        ("kaggle_qwen_ft_export.ipynb", build_export_notebook),
        ("vertex_qwen_ft_submit_and_eval.ipynb", build_submit_notebook),
        ("kaggle_qwen_ft_modal_export.ipynb", build_modal_export_notebook),
    ):
        path = NOTEBOOK_DIR / name
        path.write_text(json.dumps(builder(), indent=1) + "\n", encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
