"""Tests for the Qwen runtime boundary: dtype, device sharding, and OOM policy.

These pin decisions that change results or hide failures, so they run without
torch by injecting a stub module. Each test states which real-world failure it
guards against.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

import pytest

from aiforensics.baselines.qwen_vl import runtime as rt


class _StubDtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<dtype {self.name}>"


class _StubOutOfMemoryError(Exception):
    pass


class _NoGrad:
    """Stand-in for ``torch.no_grad()``."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _stub_module(name: str) -> types.ModuleType:
    """Build a stub module that ``importlib.util.find_spec`` accepts."""
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return module


def _install_stub_torch(monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool = True):
    """Install a minimal torch stub so runtime logic is testable without torch.

    ``__spec__`` must be set: ``load_model`` calls ``importlib.util.find_spec``
    on its dependencies, and that raises for a module without a spec.
    """
    torch = _stub_module("torch")
    torch.bfloat16 = _StubDtype("bfloat16")  # type: ignore[attr-defined]
    torch.float16 = _StubDtype("float16")  # type: ignore[attr-defined]
    torch.float32 = _StubDtype("float32")  # type: ignore[attr-defined]
    torch.no_grad = _NoGrad  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda_available,
        OutOfMemoryError=_StubOutOfMemoryError,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


class _FakeModel:
    def __init__(self, device_map: dict[str, object] | None = None) -> None:
        self.hf_device_map = device_map if device_map is not None else {}
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True


def _install_stub_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_map_result: dict[str, object] | None = None,
    record: dict | None = None,
):
    """Install a transformers stub that records how the model was requested."""
    module = _stub_module("transformers")

    class _Model:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            if record is not None:
                record["model_id"] = model_id
                record.update(kwargs)
            return _FakeModel(device_map_result)

    class _Processor:
        @staticmethod
        def from_pretrained(model_id):
            return f"processor:{model_id}"

    module.Qwen2_5_VLForConditionalGeneration = _Model  # type: ignore[attr-defined]
    module.AutoProcessor = _Processor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)

    # load_model checks these are importable before touching them.
    for name in ("qwen_vl_utils", "accelerate"):
        monkeypatch.setitem(sys.modules, name, _stub_module(name))
    return module


class TestResolveTorchDtype:
    def test_each_supported_dtype_maps_to_torch(self, monkeypatch):
        torch = _install_stub_torch(monkeypatch)
        assert rt.resolve_torch_dtype("bfloat16") is torch.bfloat16
        assert rt.resolve_torch_dtype("float16") is torch.float16
        assert rt.resolve_torch_dtype("float32") is torch.float32

    def test_unknown_dtype_raises_instead_of_defaulting(self, monkeypatch):
        """A typo must fail loudly, not silently pick a different precision."""
        _install_stub_torch(monkeypatch)
        with pytest.raises(ValueError, match="Unsupported qwen dtype"):
            rt.resolve_torch_dtype("float8")

    def test_supported_dtypes_are_exported(self):
        assert rt.SUPPORTED_DTYPES == ("bfloat16", "float16", "float32")


class TestLoadModelDtype:
    def test_configured_dtype_reaches_from_pretrained(self, monkeypatch):
        torch = _install_stub_torch(monkeypatch)
        record: dict = {}
        _install_stub_transformers(monkeypatch, device_map_result={"": 0}, record=record)

        rt.load_model("model-id", "cuda", True, RuntimeError, dtype="float16")
        assert record["torch_dtype"] is torch.float16

    def test_default_dtype_is_bfloat16(self, monkeypatch):
        torch = _install_stub_torch(monkeypatch)
        record: dict = {}
        _install_stub_transformers(monkeypatch, device_map_result={"": 0}, record=record)

        rt.load_model("model-id", "cuda", True, RuntimeError)
        assert record["torch_dtype"] is torch.bfloat16

    def test_invalid_dtype_is_reported_through_the_deferral_path(self, monkeypatch):
        _install_stub_torch(monkeypatch)
        _install_stub_transformers(monkeypatch, device_map_result={"": 0})

        class Deferred(Exception):
            pass

        with pytest.raises(Deferred, match="Unsupported qwen dtype"):
            rt.load_model("model-id", "cuda", True, Deferred, dtype="int4")


class TestLoadModelDeviceMap:
    def test_cuda_becomes_auto_so_every_gpu_is_used(self, monkeypatch):
        """A bare "cuda" map pins all weights to device 0 and wastes other GPUs."""
        _install_stub_torch(monkeypatch)
        record: dict = {}
        _install_stub_transformers(monkeypatch, device_map_result={"": 0}, record=record)

        rt.load_model("model-id", "cuda", True, RuntimeError, dtype="float16")
        assert record["device_map"] == "auto"

    def test_explicit_device_is_passed_through(self, monkeypatch):
        _install_stub_torch(monkeypatch)
        record: dict = {}
        _install_stub_transformers(monkeypatch, device_map_result={"": 0}, record=record)

        rt.load_model("model-id", "cuda:1", True, RuntimeError)
        assert record["device_map"] == "cuda:1"

    def test_model_is_put_in_eval_mode(self, monkeypatch):
        _install_stub_torch(monkeypatch)
        _install_stub_transformers(monkeypatch, device_map_result={"": 0})

        model, _processor = rt.load_model("model-id", "cuda", True, RuntimeError)
        assert model.eval_called

    @pytest.mark.parametrize("placement", ["cpu", "disk", "meta"])
    def test_offloaded_weights_are_rejected(self, monkeypatch, placement):
        """Offloading makes a run look like it hangs; refuse instead."""
        _install_stub_torch(monkeypatch)
        _install_stub_transformers(
            monkeypatch, device_map_result={"visual": 0, "model.layers.20": placement}
        )

        class Deferred(Exception):
            pass

        with pytest.raises(Deferred, match="offloaded off-GPU"):
            rt.load_model("model-id", "cuda", True, Deferred)

    def test_multi_gpu_placement_is_accepted(self, monkeypatch):
        _install_stub_torch(monkeypatch)
        _install_stub_transformers(
            monkeypatch, device_map_result={"visual": 0, "model.layers.20": 1}
        )

        model, _processor = rt.load_model("model-id", "cuda", True, RuntimeError)
        assert rt.describe_device_map(model) == {"visual": "0", "model.layers.20": "1"}

    def test_empty_device_map_is_accepted(self, monkeypatch):
        """Some stacks expose no device map; absence is not evidence of offload."""
        _install_stub_torch(monkeypatch)
        _install_stub_transformers(monkeypatch, device_map_result={})

        model, _processor = rt.load_model("model-id", "cuda", True, RuntimeError)
        assert rt.describe_device_map(model) == {}


class TestInputDevice:
    def test_inputs_follow_the_first_shard(self, monkeypatch):
        _install_stub_torch(monkeypatch)
        model = _FakeModel({"model.layers.0": 0, "visual": 1})
        assert rt._input_device(model, "cuda") == "cuda:0"

    def test_named_placement_is_used_verbatim(self, monkeypatch):
        _install_stub_torch(monkeypatch)
        model = _FakeModel({"model.layers.0": "cuda:1"})
        assert rt._input_device(model, "cuda") == "cuda:1"

    def test_missing_device_map_falls_back(self, monkeypatch):
        _install_stub_torch(monkeypatch)
        assert rt._input_device(_FakeModel({}), "cuda") == "cuda"


class TestOutOfMemoryPolicy:
    def _stub_generation(self, monkeypatch, exception: Exception):
        """Make generate_one_image reach the generation step, then raise."""
        _install_stub_torch(monkeypatch)

        utils = _stub_module("qwen_vl_utils")
        utils.process_vision_info = lambda messages: ([], [])  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qwen_vl_utils", utils)

        class _Processor:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prompt"

            def __call__(self, **kwargs):
                return types.SimpleNamespace(to=lambda device: _Inputs())

        class _Inputs(dict):
            """Mapping so ``**inputs`` works, with the attribute generation reads."""

            input_ids = []

        class _Model:
            hf_device_map: dict[str, str] = {}

            def generate(self, **kwargs):
                raise exception

        return _Model(), _Processor()

    def test_torch_oom_becomes_qwen_out_of_memory(self, monkeypatch):
        model, processor = self._stub_generation(monkeypatch, _StubOutOfMemoryError("CUDA OOM"))
        with pytest.raises(rt.QwenOutOfMemoryError, match="out of memory"):
            rt.generate_one_image(model, processor, Path("img.png"), "p", "cuda", 16)

    def test_runtime_error_mentioning_oom_is_classified_as_oom(self, monkeypatch):
        """Some stacks raise a plain RuntimeError for OOM."""
        model, processor = self._stub_generation(
            monkeypatch, RuntimeError("CUDA out of memory. Tried to allocate 5.73 GiB")
        )
        with pytest.raises(rt.QwenOutOfMemoryError):
            rt.generate_one_image(model, processor, Path("img.png"), "p", "cuda", 16)

    def test_other_errors_stay_generic_failures(self, monkeypatch):
        model, processor = self._stub_generation(monkeypatch, ValueError("bad shape"))
        with pytest.raises(Exception, match="Inference failed") as exc:
            rt.generate_one_image(model, processor, Path("img.png"), "p", "cuda", 16)
        assert not isinstance(exc.value, rt.QwenOutOfMemoryError)


class TestGenerationDeterminism:
    def test_temperature_is_not_passed_and_sampling_is_off(self, monkeypatch):
        """Greedy decoding is what makes runs reproducible; temperature is ignored."""
        _install_stub_torch(monkeypatch)

        utils = _stub_module("qwen_vl_utils")
        utils.process_vision_info = lambda messages: ([], [])  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qwen_vl_utils", utils)

        captured: dict = {}

        class _Processor:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prompt"

            def __call__(self, **kwargs):
                return types.SimpleNamespace(to=lambda device: _Inputs())

            def batch_decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
                return ["output"]

        class _Inputs(dict):
            input_ids = [[1, 2, 3]]

        class _Model:
            hf_device_map: dict[str, object] = {}

            def generate(self, **kwargs):
                captured.update(kwargs)
                return [[1, 2, 3, 4]]

        text = rt.generate_one_image(_Model(), _Processor(), Path("i.png"), "p", "cuda", 16)
        assert text == "output"
        assert captured["do_sample"] is False
        assert "temperature" not in captured


class TestShardedInputPlacement:
    """Inputs must land on the shard holding the first module, not a bare "cuda".

    With ``device_map="auto"`` the embedding layer can sit on any GPU; sending
    inputs to the wrong one raises a device-mismatch error at generation time.
    """

    def _run_with_device_map(self, monkeypatch, device_map: dict[str, object]) -> str:
        _install_stub_torch(monkeypatch)

        utils = _stub_module("qwen_vl_utils")
        utils.process_vision_info = lambda messages: ([], [])  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qwen_vl_utils", utils)

        seen: dict[str, str] = {}

        class _Inputs(dict):
            input_ids = [[1, 2]]

        class _Processor:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prompt"

            def __call__(self, **kwargs):
                def _to(device):
                    seen["device"] = device
                    return _Inputs()

                return types.SimpleNamespace(to=_to)

            def batch_decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
                return ["output"]

        class _Model:
            hf_device_map = device_map

            def generate(self, **kwargs):
                return [[1, 2, 3]]

        rt.generate_one_image(_Model(), _Processor(), Path("i.png"), "p", "cuda", 8)
        return seen["device"]

    def test_inputs_go_to_the_first_shard_not_bare_cuda(self, monkeypatch):
        device = self._run_with_device_map(monkeypatch, {"model.embed_tokens": 1, "visual": 0})
        assert device == "cuda:1"

    def test_bare_cuda_used_when_no_device_map_exists(self, monkeypatch):
        assert self._run_with_device_map(monkeypatch, {}) == "cuda"
