# Qwen Fine-Tuning On Modal

This runbook operates the `qwen_ft` pipeline using Modal: exporting a balanced GenImage subset zip from Kaggle, uploading it to Modal Volume `aiforensics-qwen-ft`, fine-tuning LoRA on `Qwen/Qwen2.5-VL-7B-Instruct` using an A100-40GB GPU, and running evaluation through the `aiforensics` CLI.

Design spec: `docs/superpowers/specs/2026-09-05-qwen-finetune-modal-pivot.md`.
Implementation plan: `docs/superpowers/plans/2026-09-05-qwen-finetune-modal-pivot.md`.

## One-Time Setup

Ensure the Modal CLI is authenticated and the volume and secret exist:

```bash
python3 -m modal profile current
python3 -m modal volume list
python3 -m modal secret list
```

Expected:

- profile is `tnhatnguyen-dev2805`
- volume includes `aiforensics-qwen-ft`
- secret includes `huggingface-secret`

If the volume does not exist, create it:

```bash
python3 -m modal volume create aiforensics-qwen-ft
```

If the Hugging Face secret does not exist, create it:

```bash
python3 -m modal secret create huggingface-secret HF_TOKEN=<YOUR_HF_TOKEN>
```

## Export Data On Kaggle

Open `notebooks/kaggle_qwen_ft_modal_export.ipynb` on Kaggle:

1. Attach the GenImage dataset (layout `<DATA_ROOT>/<generator>/<train|val>/{ai,nature}/*`).
2. Run all cells in order.
3. Save the notebook with output enabled and download the exported zip:

```text
/kaggle/working/qwen_ft_protocol_a_small.zip
```

Protocol A defaults: 100 train and 50 eval images per label per generator (7 generators), seed 70.

## Upload Zip To Modal Volume

Upload the downloaded archive into the volume's `/archives` directory:

```bash
python3 -m modal volume put aiforensics-qwen-ft \
  ~/aiforensics-modal-data/qwen_ft_protocol_a_small.zip \
  /archives/qwen_ft_protocol_a_small.zip
```

## Inspect And Unpack

Verify the archive is present on the volume, then extract it to `/vol/data/protocol-a-small`:

```bash
python3 -m modal run infra/modal/qwen_ft_modal.py::inspect_volume
python3 -m modal run infra/modal/qwen_ft_modal.py::unpack_archive
```

This extracts images and manifests and prepares `qwen_train_abs.jsonl` with absolute volume paths for training.

## Train LoRA

Run LoRA fine-tuning on an A100-40GB GPU:

```bash
python3 -m modal run infra/modal/qwen_ft_modal.py::train_lora
```

The final adapter will be saved to `/vol/checkpoints/protocol-a-small/final_adapter`.

## Eval Small

Evaluate the trained adapter on the 700-image eval-small subset:

```bash
python3 -m modal run infra/modal/qwen_ft_modal.py::eval_small
```

This generates `predictions.jsonl`, `metrics.json`, `metrics_by_source.csv`, and a Markdown report under `/vol/eval/protocol-a-small/eval-small`.

## Eval Full (Optional)

To evaluate on the full 7,000-image GenImage validation set:
1. Export the archive on Kaggle using `--include-eval-full` (or `--eval-full-per-label 500`), which includes `eval-full/` images and `manifests/eval_full.csv`.
2. Upload and unpack the archive to the volume.
3. Run full evaluation:

```bash
python3 -m modal run infra/modal/qwen_ft_modal.py::eval_full
```

## Download Results

Download evaluation artifacts to your local machine:

```bash
python3 -m modal volume get aiforensics-qwen-ft \
  /eval/protocol-a-small/eval-small \
  ~/aiforensics-modal-results/eval-small
```
