import pytest

from aiforensics.schemas.predictions import (
    PredictionError,
    PredictionRecord,
    validate_prediction_record,
)


def test_validate_prediction_record_valid_clip():
    record_dict = {
        "sample_id": "smoke/dev/real_0002",
        "label_true": "real",
        "label_pred": "real",
        "score_fake": 0.12,
        "model_name": "clip_probe",
        "source": "smoke",
        "dataset": "smoke",
        "split": "dev",
        "checksum": "a" * 64,
    }
    record = validate_prediction_record(record_dict)
    assert record.sample_id == "smoke/dev/real_0002"
    assert record.model_name == "clip_probe"
    assert record.score_fake == 0.12


def test_validate_prediction_record_accepts_score_fake_none():
    record_dict = {
        "sample_id": "test_id",
        "label_true": "real",
        "label_pred": "unknown",
        "score_fake": None,
        "model_name": "clip_probe",
        "source": "smoke",
    }
    record = validate_prediction_record(record_dict)
    assert record.score_fake is None


def test_validate_prediction_record_rejects_missing_sample_id():
    record_dict = {
        "label_true": "real",
        "label_pred": "real",
        "score_fake": 0.5,
        "model_name": "clip_probe",
        "source": "smoke",
    }
    with pytest.raises(PredictionError, match="Field 'sample_id'"):
        validate_prediction_record(record_dict)


def test_validate_prediction_record_rejects_invalid_label_true():
    record_dict = {
        "sample_id": "test",
        "label_true": "invalid",
        "label_pred": "real",
        "score_fake": 0.5,
        "model_name": "clip_probe",
        "source": "smoke",
    }
    with pytest.raises(PredictionError, match="Field 'label_true'"):
        validate_prediction_record(record_dict)


def test_validate_prediction_record_rejects_invalid_label_pred():
    record_dict = {
        "sample_id": "test",
        "label_true": "real",
        "label_pred": "invalid",
        "score_fake": 0.5,
        "model_name": "clip_probe",
        "source": "smoke",
    }
    with pytest.raises(PredictionError, match="Field 'label_pred'"):
        validate_prediction_record(record_dict)


def test_validate_prediction_record_rejects_score_fake_less_than_zero():
    record_dict = {
        "sample_id": "test",
        "label_true": "real",
        "label_pred": "real",
        "score_fake": -0.1,
        "model_name": "clip_probe",
        "source": "smoke",
    }
    with pytest.raises(PredictionError, match="Field 'score_fake'"):
        validate_prediction_record(record_dict)


def test_validate_prediction_record_rejects_score_fake_greater_than_one():
    record_dict = {
        "sample_id": "test",
        "label_true": "real",
        "label_pred": "real",
        "score_fake": 1.1,
        "model_name": "clip_probe",
        "source": "smoke",
    }
    with pytest.raises(PredictionError, match="Field 'score_fake'"):
        validate_prediction_record(record_dict)


def test_validate_prediction_record_rejects_invalid_model_name():
    record_dict = {
        "sample_id": "test",
        "label_true": "real",
        "label_pred": "real",
        "score_fake": 0.5,
        "model_name": "invalid",
        "source": "smoke",
    }
    with pytest.raises(PredictionError, match="Field 'model_name'"):
        validate_prediction_record(record_dict)


def test_write_and_load_roundtrip_jsonl(tmp_path):
    records = [
        PredictionRecord(
            sample_id="id1",
            label_true="fake",
            label_pred="fake",
            score_fake=0.9,
            model_name="qwen_vl",
            source="src",
            parse_status="parsed",
        ),
        PredictionRecord(
            sample_id="id2",
            label_true="real",
            label_pred="real",
            score_fake=None,
            model_name="clip_probe",
            source="src",
        ),
    ]
    out_path = tmp_path / "preds.jsonl"

    from aiforensics.schemas.predictions import load_predictions, write_predictions

    write_predictions(records, out_path)

    loaded = load_predictions(out_path)
    assert len(loaded) == 2
    assert loaded[0].sample_id == "id1"
    assert loaded[0].label_true == "fake"
    assert loaded[1].sample_id == "id2"
    assert loaded[1].score_fake is None

    # ensure it wrote jsonl and excluded None
    raw_lines = out_path.read_text().splitlines()
    assert len(raw_lines) == 2
    import json

    parsed0 = json.loads(raw_lines[0])
    parsed1 = json.loads(raw_lines[1])
    assert "parse_status" in parsed0
    assert "parse_status" not in parsed1


def test_load_predictions_missing_file(tmp_path):
    from aiforensics.schemas.predictions import load_predictions

    with pytest.raises(PredictionError, match="missing"):
        load_predictions(tmp_path / "does_not_exist.jsonl")


def test_load_predictions_invalid_json(tmp_path):
    from aiforensics.schemas.predictions import load_predictions

    out_path = tmp_path / "bad.jsonl"
    ok_json = (
        '{"sample_id":"id1","label_true":"real","label_pred":"real",'
        '"score_fake":0.5,"model_name":"clip_probe","source":"s"}'
    )
    out_path.write_text(f"{ok_json}\n{{bad_json}}\n")
    with pytest.raises(PredictionError, match="Invalid JSON on line 2 in"):
        load_predictions(out_path)


def test_load_predictions_invalid_record_data(tmp_path):
    from aiforensics.schemas.predictions import load_predictions

    out_path = tmp_path / "bad_data.jsonl"
    # line 1 is ok, line 2 is bad
    ok_json = (
        '{"sample_id":"id1","label_true":"real","label_pred":"real",'
        '"score_fake":0.5,"model_name":"clip_probe","source":"s"}'
    )
    bad_json = (
        '{"sample_id":"id2","label_true":"wrong_label","label_pred":"real",'
        '"score_fake":0.5,"model_name":"clip_probe","source":"s"}'
    )
    out_path.write_text(f"{ok_json}\n{bad_json}\n")
    with pytest.raises(PredictionError, match="line 2"):
        load_predictions(out_path)

    record_dict = {
        "sample_id": "test",
        "label_true": "real",
        "label_pred": "real",
        "score_fake": 0.5,
        "model_name": "clip_probe",
        "source": "smoke",
        "checksum": "invalid_hex",
    }
    with pytest.raises(PredictionError, match="Field 'checksum'"):
        validate_prediction_record(record_dict)


def test_validate_predictions():
    from aiforensics.schemas.predictions import PredictionRecord, validate_predictions

    records = [
        PredictionRecord(
            sample_id="id1",
            label_true="real",
            label_pred="real",
            score_fake=0.1,
            model_name="clip_probe",
            source="s1",
        ),
        PredictionRecord(
            sample_id="id1",  # duplicate
            label_true="fake",
            label_pred="fake",
            score_fake=0.9,
            model_name="qwen_vl",
            source="s2",
            # missing prompt_id, raw_output, explanation, parse_status for qwen_vl
        ),
        PredictionRecord(
            sample_id="id2",
            label_true="real",
            label_pred="real",
            score_fake=0.1,
            model_name="clip_probe",
            source="s1",
            parse_status="parsed",  # Invalid for non-mllm
        ),
        PredictionRecord(
            sample_id="id3",
            label_true="fake",
            label_pred="fake",
            score_fake=0.9,
            model_name="assisted_qwen",
            source="s1",
            prompt_id="p1",
            raw_output="raw",
            explanation="exp",
            parse_status="parsed",
        ),
    ]

    manifest_ids = {"id1", "id2", "id3", "id4"}

    result = validate_predictions(
        records, manifest_sample_ids=manifest_ids, require_mllm_fields=True
    )

    assert not result.is_valid
    assert result.total_records == 4

    # Counts
    assert result.records_by_model == {"clip_probe": 2, "qwen_vl": 1, "assisted_qwen": 1}
    assert result.records_by_label_true == {"real": 2, "fake": 2}
    assert result.records_by_label_pred == {"real": 2, "fake": 2}
    assert result.records_by_source == {"s1": 3, "s2": 1}

    # Duplicates
    assert "id1" in result.duplicate_sample_ids

    # MLLM missing fields
    assert any("qwen_vl" in err and "MLLM fields" in err for err in result.errors)

    # Non-MLLM parse_status
    assert any("clip_probe" in err and "parse_status" in err for err in result.errors)


def test_validate_predictions_manifest_ids():
    from aiforensics.schemas.predictions import PredictionRecord, validate_predictions

    records = [
        PredictionRecord(
            sample_id="id_not_manifest",
            label_true="real",
            label_pred="real",
            score_fake=0.1,
            model_name="clip_probe",
            source="s1",
        )
    ]
    manifest_ids = {"id1"}
    result = validate_predictions(records, manifest_sample_ids=manifest_ids)
    assert not result.is_valid
    assert "id_not_manifest" in result.missing_manifest_sample_ids
    assert any("not found in manifest" in err for err in result.errors)


def test_validate_prediction_record_rejects_missing_score_fake():
    record_dict = {
        "sample_id": "test",
        "label_true": "real",
        "label_pred": "real",
        "model_name": "clip_probe",
        "source": "smoke",
    }
    with pytest.raises(PredictionError, match="Field 'score_fake'"):
        validate_prediction_record(record_dict)


def test_load_predictions_blank_line_raises_error(tmp_path):
    from aiforensics.schemas.predictions import load_predictions

    out_path = tmp_path / "blank.jsonl"
    # A file with a blank line in the middle
    ok_json = (
        '{"sample_id":"id1","label_true":"real","label_pred":"real",'
        '"score_fake":0.5,"model_name":"clip_probe","source":"s"}'
    )
    out_path.write_text(f"{ok_json}\n\n{ok_json}\n")
    with pytest.raises(PredictionError, match="blank line not allowed"):
        load_predictions(out_path)


def test_load_predictions_empty_file_returns_empty_list(tmp_path):
    from aiforensics.schemas.predictions import load_predictions

    out_path = tmp_path / "empty.jsonl"
    out_path.write_text("")
    assert load_predictions(out_path) == []
