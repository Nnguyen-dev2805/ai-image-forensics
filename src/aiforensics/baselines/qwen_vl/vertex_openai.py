"""Vertex AI OpenAI-compatible runtime for Qwen-VL baselines."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VertexQwenError(RuntimeError):
    """Raised when the Vertex OpenAI-compatible runtime cannot be used."""


def build_vertex_base_url(
    *,
    project_id: str,
    location: str,
    endpoint_id: str,
    endpoint_domain: str,
) -> str:
    cleaned_domain = endpoint_domain.removeprefix("https://").rstrip("/")
    if cleaned_domain == "aiplatform.googleapis.com":
        raise ValueError("Dedicated Endpoint runs must use the prediction.vertexai.goog domain")
    if not cleaned_domain.endswith(".prediction.vertexai.goog"):
        raise ValueError(f"Unexpected Vertex dedicated endpoint domain: {cleaned_domain}")
    return (
        f"https://{cleaned_domain}/v1/projects/{project_id}"
        f"/locations/{location}/endpoints/{endpoint_id}"
    )


def create_vertex_openai_client(
    *,
    project_id: str | None,
    location: str | None,
    endpoint_id: str | None,
    endpoint_domain: str | None,
    credentials_env_var: str,
) -> Any:
    missing = [
        name
        for name, value in {
            "vertex_project_id": project_id,
            "vertex_location": location,
            "vertex_endpoint_id": endpoint_id,
            "vertex_endpoint_domain": endpoint_domain,
        }.items()
        if not value
    ]
    if missing:
        raise VertexQwenError(f"Missing Vertex Qwen config field(s): {', '.join(missing)}")

    service_account_json = os.environ.get(credentials_env_var)
    if not service_account_json:
        raise VertexQwenError(
            f"Missing service-account JSON in environment variable {credentials_env_var!r}"
        )

    try:
        service_account_info = json.loads(service_account_json)
    except json.JSONDecodeError as exc:
        raise VertexQwenError(f"Invalid service-account JSON in {credentials_env_var!r}") from exc

    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        from openai import OpenAI
    except ImportError as exc:
        raise VertexQwenError(
            "Missing Vertex Qwen dependencies: install the 'vertex' optional dependency group"
        ) from exc

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    credentials.refresh(Request())
    base_url = build_vertex_base_url(
        project_id=project_id or "",
        location=location or "",
        endpoint_id=endpoint_id or "",
        endpoint_domain=endpoint_domain or "",
    )
    client = OpenAI(api_key=credentials.token, base_url=base_url, max_retries=5)
    client._vertex_credentials = credentials
    return client


def image_data_url(image_path: Path) -> str:
    mime_type, _encoding = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def generate_one_image_via_vertex(
    *,
    client: Any,
    model_id: str | None,
    image_path: Path,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if not model_id:
        raise VertexQwenError("Missing Vertex Qwen model id")

    credentials = getattr(client, "_vertex_credentials", None)
    if credentials is not None:
        try:
            from google.auth.transport.requests import Request

            if not credentials.valid:
                credentials.refresh(Request())
                client.api_key = credentials.token
        except Exception as e:
            logger.warning("Could not pre-refresh Vertex credentials: %s", e)

    def _call():
        return client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                    ],
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        response = _call()
    except Exception as exc:
        err_msg = str(exc).lower()
        if (
            "401" in err_msg or "unauthorized" in err_msg or "authentication" in err_msg
        ) and credentials is not None:
            try:
                from google.auth.transport.requests import Request

                credentials.refresh(Request())
                client.api_key = credentials.token
                response = _call()
            except Exception as refresh_exc:
                raise exc from refresh_exc
        else:
            raise

    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise VertexQwenError("Vertex Qwen response did not contain text content")
    return content
