from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ProjectConfig(BaseModel):
    name: str
    phase: str
    description: str


class PathsConfig(BaseModel):
    data_root: Path
    manifest_root: Path
    cache_root: Path
    output_root: Path
    external_root: Path


class RuntimeConfig(BaseModel):
    python: str
    seed: int
    device: str
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    fail_fast: bool


class TinyGenImageConfig(BaseModel):
    enabled: bool
    source: str
    use_original_split: bool
    train_manifest: Path
    dev_manifest: Path
    # GenImage-layout generator directories used as in-distribution data.
    generators: list[str] = Field(default_factory=list)
    max_images: int = Field(default=0, ge=0)
    balance_labels: bool = True


class GenImageUnseenConfig(BaseModel):
    enabled: bool
    # Held-out GenImage-layout generator directories, never used for training.
    generators: list[str] = Field(default_factory=list)
    max_images: int = Field(ge=0)
    balance_labels: bool
    split: str
    manifest: Path
    # Optional on-disk split to collect from (e.g. "val"); defaults to all splits.
    source_split: str | None = None


class SynthbusterConfig(BaseModel):
    enabled: bool
    max_images: int
    balance_labels: bool
    split: str
    manifest: Path


class DatasetsConfig(BaseModel):
    tiny_genimage: TinyGenImageConfig
    genimage_unseen: GenImageUnseenConfig
    synthbuster: SynthbusterConfig


class ClipProbeConfig(BaseModel):
    enabled: bool
    model_family: Literal["synthetic", "openclip"]
    model_name: str
    pretrained: str
    classifier: str
    seeds: list[int]
    cache_embeddings: bool

    @model_validator(mode="after")
    def validate_seeds(self) -> "ClipProbeConfig":
        if self.enabled and not self.seeds:
            raise ValueError("clip_probe.seeds must not be empty when enabled is true")
        return self


class QwenVLConfig(BaseModel):
    enabled: bool
    model_id: str
    provider: Literal["local", "vertex_openai"] = "local"
    prompt_id: str
    temperature: float = Field(ge=0.0)
    max_new_tokens: int = Field(gt=0)
    cache_outputs: bool
    allow_deferred: bool = True
    # Compute dtype changes numerical results, so it is configured and recorded
    # (cache keys, reports) instead of hidden in the runtime.
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    vertex_project_id: str | None = None
    vertex_location: str | None = None
    vertex_endpoint_id: str | None = None
    vertex_endpoint_domain: str | None = None
    vertex_model_id: str | None = None
    vertex_credentials_env_var: str = "AIF_GOOGLE_APPLICATION_CREDENTIALS_JSON"


class AssistedQwenConfig(BaseModel):
    enabled: bool
    base_model_id: str
    provider: Literal["local", "vertex_openai"] = "local"
    prompt_id: str
    assistant_source: str
    include_classifier_pred: bool
    include_fake_probability: bool
    temperature: float = Field(ge=0.0)
    max_new_tokens: int = Field(gt=0)
    cache_outputs: bool
    allow_deferred: bool = True
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    vertex_project_id: str | None = None
    vertex_location: str | None = None
    vertex_endpoint_id: str | None = None
    vertex_endpoint_domain: str | None = None
    vertex_model_id: str | None = None
    vertex_credentials_env_var: str = "AIF_GOOGLE_APPLICATION_CREDENTIALS_JSON"


class NPRConfig(BaseModel):
    enabled: bool
    repo_url: str
    repo_commit: str | None
    checkpoint_path: Path
    checkpoint_sha256: str | None
    batch_size: int = Field(gt=0)
    allow_deferred: bool


class QwenFTConfig(BaseModel):
    enabled: bool = False
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # GCS URI of the trained LoRA adapter; empty until a Vertex job finishes.
    adapter_uri: str = ""
    prompt_id: str = "qwen_ft_label_json_v1"
    temperature: float = Field(default=0.0, ge=0.0)
    max_new_tokens: int = Field(default=32, gt=0)
    cache_outputs: bool = True
    allow_deferred: bool = True
    dtype: Literal["bfloat16", "float16", "float32"] = "float16"
    # Label-only training target: no confidence, no evidence in this stage.
    output_fields: list[Literal["label"]] = ["label"]
    min_pixels: int = Field(default=50176, gt=0)
    max_pixels: int = Field(default=200704, gt=0)

    @model_validator(mode="after")
    def validate_qwen_ft_config(self) -> "QwenFTConfig":
        if self.min_pixels > self.max_pixels:
            raise ValueError(
                f"min_pixels ({self.min_pixels}) must be <= max_pixels ({self.max_pixels})"
            )
        if self.enabled and not self.adapter_uri:
            raise ValueError("baselines.qwen_ft.adapter_uri is required when qwen_ft is enabled")
        return self


class BaselinesConfig(BaseModel):
    clip_probe: ClipProbeConfig
    qwen_vl: QwenVLConfig
    assisted_qwen: AssistedQwenConfig
    npr: NPRConfig
    # Optional with an off default so pre-qwen_ft configs keep loading unchanged.
    qwen_ft: QwenFTConfig = Field(default_factory=QwenFTConfig)


class LabelsConfig(BaseModel):
    negative: str
    positive: str

    @model_validator(mode="after")
    def validate_labels(self) -> "LabelsConfig":
        if self.negative != "real":
            raise ValueError("evaluation.labels.negative must be 'real'")
        if self.positive != "fake":
            raise ValueError("evaluation.labels.positive must be 'fake'")
        return self


class EvaluationConfig(BaseModel):
    labels: LabelsConfig
    metrics: list[str]
    group_by: list[str]


class ReportConfig(BaseModel):
    filename: str
    include_failure_notes: bool
    include_explanations_sample: bool
    explanation_sample_size: int = Field(ge=0)


class AppConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    runtime: RuntimeConfig
    datasets: DatasetsConfig
    baselines: BaselinesConfig
    evaluation: EvaluationConfig
    report: ReportConfig
