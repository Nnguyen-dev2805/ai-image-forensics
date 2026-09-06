#!/usr/bin/env python
"""Submit the qwen_ft Vertex AI CustomJob from explicit flags.

Builds the CustomJob REST payload via ``build_vertex_custom_job_spec`` and
either prints it (``--dry-run``) or submits it to the Vertex AI API with
Application Default Credentials. The trainer image defaults to the project's
Artifact Registry repository that the Task 5 container recipe pushes to.
"""

from __future__ import annotations

import argparse
import json

from aiforensics.finetune.vertex_job import (
    DEFAULT_MODEL_ID,
    build_vertex_custom_job_spec,
)

DEFAULT_BUCKET_URI = "gs://aiforensics-qwen-ft-579187260419"
DEFAULT_LOCATION = "asia-southeast1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit the qwen_ft Vertex CustomJob.")
    parser.add_argument("--project-id", required=True, help="GCP project id or number")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="Vertex region")
    parser.add_argument("--bucket-uri", default=DEFAULT_BUCKET_URI)
    parser.add_argument(
        "--protocol",
        default="protocol-a-small",
        help="Protocol name; selects GCS checkpoint and data prefixes",
    )
    parser.add_argument(
        "--training-jsonl-uri",
        default=None,
        help="Defaults to <bucket>/data/<protocol>/qwen_train.jsonl",
    )
    parser.add_argument(
        "--image-uri",
        default=None,
        help="Defaults to the project Artifact Registry trainer image",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--machine-type", default="a2-highgpu-1g")
    parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_A100")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the CustomJob JSON payload without submitting",
    )
    return parser


def submit(project_id: str, location: str, spec: dict) -> dict:
    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
    except ImportError as exc:
        raise SystemExit(
            "Missing Vertex dependencies: install the 'vertex' optional dependency "
            "group (uv sync --extra vertex) or pip install google-auth"
        ) from exc

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(credentials)
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/customJobs"
    )
    response = session.post(url, json=spec)
    if response.status_code != 200:
        raise SystemExit(
            f"Vertex CustomJob submission failed ({response.status_code}): {response.text}"
        )
    return response.json()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    training_jsonl_uri = args.training_jsonl_uri or (
        f"{args.bucket_uri.rstrip('/')}/data/{args.protocol}/qwen_train.jsonl"
    )

    spec = build_vertex_custom_job_spec(
        project_id=args.project_id,
        location=args.location,
        bucket_uri=args.bucket_uri,
        protocol=args.protocol,
        training_jsonl_uri=training_jsonl_uri,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        model_id=args.model_id,
        image_uri=args.image_uri,
    )

    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return 0

    job = submit(args.project_id, args.location, spec)
    print(f"[submit] submitted CustomJob: {job.get('name', '<no name>')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
