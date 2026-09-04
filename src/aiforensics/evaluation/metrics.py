import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from aiforensics.runs.scope import RunScope, scope_matches
from aiforensics.schemas.predictions import (
    PredictionRecord,
    load_predictions,
    validate_predictions,
)


class MetricsError(ValueError):
    """Exception raised for errors during metrics computation or validation."""

    pass


METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
)


def compute_classification_metrics(
    records: Sequence[PredictionRecord],
) -> dict[str, float | None]:
    if not records:
        return {k: None for k in METRIC_NAMES}

    tp = fp = tn = fn = 0
    true_real_count = 0
    true_fake_count = 0

    scores = []
    y_true = []

    for r in records:
        # Count totals for true classes
        if r.label_true == "real":
            true_real_count += 1
        elif r.label_true == "fake":
            true_fake_count += 1

        # Calculate exact matches (confusion matrix elements)
        # Note: unknown is always incorrect, so it doesn't add to TN/TP/FP/FN in a way that helps,
        # but for precision/recall definitions in spec:
        # TP = true fake, pred fake
        # FP = true real, pred fake
        # TN = true real, pred real
        # FN = true fake, pred != fake
        if r.label_true == "fake" and r.label_pred == "fake":
            tp += 1
        elif r.label_true == "real" and r.label_pred == "fake":
            fp += 1
        elif r.label_true == "real" and r.label_pred == "real":
            tn += 1
        elif r.label_true == "fake" and r.label_pred != "fake":
            fn += 1

        # Collect score_fake for AUROC
        if r.score_fake is not None:
            scores.append(r.score_fake)
            y_true.append(1 if r.label_true == "fake" else 0)

    # ACCURACY
    # Note: If label_true="real" and label_pred="unknown", they don't fall into the 4 buckets above.
    # Accuracy is exact label match rate:
    correct = sum(1 for r in records if r.label_true == r.label_pred)
    accuracy = correct / len(records)

    # BALANCED ACCURACY
    real_recall = tn / true_real_count if true_real_count > 0 else None
    fake_recall = tp / (tp + fn) if (tp + fn) > 0 else None

    if real_recall is not None and fake_recall is not None:
        balanced_accuracy = (real_recall + fake_recall) / 2
    else:
        balanced_accuracy = None

    # PRECISION
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = None

    # RECALL
    recall = fake_recall

    # F1
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = None

    # AUROC
    auroc = None
    if len(scores) > 0:
        # Check if both classes are present in the scored subset
        if sum(y_true) > 0 and sum(y_true) < len(y_true):
            auroc = float(roc_auc_score(y_true, scores))

    return {
        "accuracy": float(accuracy) if accuracy is not None else None,
        "balanced_accuracy": float(balanced_accuracy) if balanced_accuracy is not None else None,
        "precision": float(precision) if precision is not None else None,
        "recall": float(recall) if recall is not None else None,
        "f1": float(f1) if f1 is not None else None,
        "auroc": auroc,
    }


def compute_metrics_by_source(
    records: Sequence[PredictionRecord],
) -> pd.DataFrame:
    columns = ["source", "n"] + list(METRIC_NAMES)

    if not records:
        return pd.DataFrame(columns=columns)

    grouped = {}
    for r in records:
        if r.source not in grouped:
            grouped[r.source] = []
        grouped[r.source].append(r)

    rows = []
    # Sorted ascending by source
    for source in sorted(grouped.keys()):
        source_records = grouped[source]
        metrics = compute_classification_metrics(source_records)
        row = {
            "source": source,
            "n": len(source_records),
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    return df


def write_metrics_outputs(
    records: Sequence[PredictionRecord],
    output_dir: Path | str,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "metrics.json"
    csv_path = out_dir / "metrics_by_source.csv"

    overall_metrics = compute_classification_metrics(records)
    df_source = compute_metrics_by_source(records)

    out_json = {"total_records": len(records), "overall": overall_metrics}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    df_source.to_csv(csv_path, index=False)

    return json_path, csv_path


def evaluate_prediction_file(
    prediction_path: Path | str,
    output_dir: Path | str | None = None,
    *,
    manifest_sample_ids: set[str] | None = None,
) -> tuple[Path, Path]:
    p_path = Path(prediction_path)
    if not p_path.is_file():
        raise MetricsError(f"File not found: {p_path}")

    # load predictions
    from aiforensics.schemas.predictions import PredictionError

    try:
        records = load_predictions(p_path)
    except PredictionError as e:
        raise MetricsError(f"Failed to load predictions: {e}") from e

    # validate
    res = validate_predictions(records, manifest_sample_ids=manifest_sample_ids)
    if not res.is_valid:
        raise MetricsError(f"Prediction validation failed for {p_path}: {res.errors}")

    out_dir = Path(output_dir) if output_dir else p_path.parent
    return write_metrics_outputs(records, out_dir)


def discover_prediction_files(output_root: Path | str) -> list[Path]:
    root = Path(output_root)
    if not root.is_dir():
        return []

    # Needs to find predictions.jsonl files
    files = list(root.rglob("predictions.jsonl"))

    return sorted(files)


def discover_scoped_prediction_files(
    output_root: Path | str,
    expected_scope: RunScope,
) -> tuple[list[Path], list[Path]]:
    """Split discovered prediction files into current-scope and out-of-scope.

    Metrics belong to the experiment that produced them, so evaluation must not
    treat every ``predictions.jsonl`` under a shared ``output_root`` as part of
    the current config. Files whose run directory does not carry the current
    run scope are returned separately so callers can report them instead of
    evaluating them.
    """
    in_scope: list[Path] = []
    out_of_scope: list[Path] = []
    for path in discover_prediction_files(output_root):
        if scope_matches(path.parent, expected_scope):
            in_scope.append(path)
        else:
            out_of_scope.append(path)
    return in_scope, out_of_scope
