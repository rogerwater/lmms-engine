from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.models.nanovlm.monkey_patch import apply_liger_rmsnorm_to_nanovlm


class DummyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = 1e-6


class DummyDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = DummyRMSNorm()
        self.post_attention_layernorm = DummyRMSNorm()


class DummyQwen3Model(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(DummyDecoderLayer() for _ in range(num_layers))
        self.norm = DummyRMSNorm()


class DummyQwen3ForCausalLM(nn.Module):
    base_model_prefix = "model"

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = DummyQwen3Model()


class DummyNanoVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="nanovlm")
        self.language_model = DummyQwen3ForCausalLM()


def fake_liger_patcher(
    module: nn.Module,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = True,
) -> None:
    module.offset = offset
    module.casting_mode = casting_mode
    module.in_place = in_place


def test_nanovlm_liger_patch_is_registered() -> None:
    assert "nanovlm" in MONKEY_PATCHER
    assert MONKEY_PATCHER["nanovlm"]["liger"] is apply_liger_rmsnorm_to_nanovlm


def test_nanovlm_liger_patch_only_updates_qwen3_rmsnorm_instances() -> None:
    model = DummyNanoVLM()
    state_dict_keys_before = tuple(model.state_dict().keys())
    parameter_ids_before = {name: id(parameter) for name, parameter in model.named_parameters()}

    with patch(
        "lmms_engine.models.nanovlm.monkey_patch._load_liger_rms_norm_patcher",
        return_value=fake_liger_patcher,
    ):
        newly_patched = apply_liger_rmsnorm_to_nanovlm(
            model=model,
            rms_norm=True,
            rms_norm_in_place=True,
            strict=True,
        )

    assert newly_patched == 5
    assert tuple(model.state_dict().keys()) == state_dict_keys_before
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids_before

    norms = [model.language_model.model.norm]
    for layer in model.language_model.model.layers:
        norms.extend((layer.input_layernorm, layer.post_attention_layernorm))

    assert all(norm.in_place is True for norm in norms)
    assert all(norm.casting_mode == "llama" for norm in norms)
    assert all(norm._lmms_engine_liger_rms_norm is True for norm in norms)


def test_nanovlm_liger_patch_is_idempotent_and_can_update_in_place_setting() -> None:
    model = DummyNanoVLM()

    with patch(
        "lmms_engine.models.nanovlm.monkey_patch._load_liger_rms_norm_patcher",
        return_value=fake_liger_patcher,
    ):
        first_count = apply_liger_rmsnorm_to_nanovlm(model=model, rms_norm_in_place=True)
        second_count = apply_liger_rmsnorm_to_nanovlm(model=model, rms_norm_in_place=False)

    assert first_count == 5
    assert second_count == 0
    assert model.language_model.model.norm.in_place is False
    for layer in model.language_model.model.layers:
        assert layer.input_layernorm.in_place is False
        assert layer.post_attention_layernorm.in_place is False


def test_nanovlm_liger_patch_rejects_unvalidated_kernels() -> None:
    model = DummyNanoVLM()

    try:
        apply_liger_rmsnorm_to_nanovlm(model=model, swiglu=True)
    except ValueError as exc:
        assert "supports RMSNorm only" in str(exc)
    else:
        raise AssertionError("Expected the NanoVLM Liger patch to reject SwiGLU")
