from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.models.nanovlm.monkey_patch import (
    apply_liger_rmsnorm_to_nanovlm,
    apply_torch_npu_rmsnorm_to_nanovlm,
)


class DummyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = 1e-6


class DummySelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_norm = DummyRMSNorm()
        self.k_norm = DummyRMSNorm()


class DummyDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = DummyRMSNorm()
        self.post_attention_layernorm = DummyRMSNorm()
        self.self_attn = DummySelfAttention()


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
        self.config = SimpleNamespace(
            model_type="nanovlm",
            text_config=SimpleNamespace(model_type="qwen3"),
        )
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


def fake_torch_npu_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del eps
    output = hidden_states * weight
    reciprocal_rms = torch.ones_like(hidden_states[..., :1])
    return output, reciprocal_rms


def get_all_qwen3_norms(model: DummyNanoVLM) -> list[DummyRMSNorm]:
    norms = [model.language_model.model.norm]
    for layer in model.language_model.model.layers:
        norms.extend(
            (
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
            )
        )
    return norms


def test_nanovlm_liger_patch_is_registered() -> None:
    assert "nanovlm" in MONKEY_PATCHER
    assert MONKEY_PATCHER["nanovlm"]["liger"] is apply_liger_rmsnorm_to_nanovlm


def test_nanovlm_torch_npu_rmsnorm_patch_is_registered() -> None:
    assert "nanovlm" in MONKEY_PATCHER
    assert (
        MONKEY_PATCHER["nanovlm"]["npu_rms_norm"]
        is apply_torch_npu_rmsnorm_to_nanovlm
    )


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

    assert newly_patched == 9
    assert tuple(model.state_dict().keys()) == state_dict_keys_before
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids_before

    norms = get_all_qwen3_norms(model)

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

    assert first_count == 9
    assert second_count == 0
    assert all(norm.in_place is False for norm in get_all_qwen3_norms(model))


def test_nanovlm_liger_patch_rejects_unvalidated_kernels() -> None:
    model = DummyNanoVLM()

    try:
        apply_liger_rmsnorm_to_nanovlm(model=model, swiglu=True)
    except ValueError as exc:
        assert "supports RMSNorm only" in str(exc)
    else:
        raise AssertionError("Expected the NanoVLM Liger patch to reject SwiGLU")


def test_nanovlm_torch_npu_patch_preserves_parameters_and_state_dict() -> None:
    model = DummyNanoVLM()
    state_dict_keys_before = tuple(model.state_dict().keys())
    parameter_ids_before = {name: id(parameter) for name, parameter in model.named_parameters()}

    with patch(
        "lmms_engine.models.nanovlm.monkey_patch._load_torch_npu_rms_norm_operator",
        return_value=fake_torch_npu_rms_norm,
    ):
        newly_patched = apply_torch_npu_rmsnorm_to_nanovlm(model=model, strict=True)

    assert newly_patched == 9
    assert tuple(model.state_dict().keys()) == state_dict_keys_before
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids_before

    input_tensor = torch.randn(2, 3, 8)
    for norm in get_all_qwen3_norms(model):
        torch.testing.assert_close(norm(input_tensor), input_tensor * norm.weight)
        assert norm._lmms_engine_torch_npu_rms_norm is True
        assert norm._lmms_engine_rms_norm_backend == "torch_npu"


def test_nanovlm_torch_npu_patch_is_idempotent() -> None:
    model = DummyNanoVLM()

    with patch(
        "lmms_engine.models.nanovlm.monkey_patch._load_torch_npu_rms_norm_operator",
        return_value=fake_torch_npu_rms_norm,
    ):
        first_count = apply_torch_npu_rmsnorm_to_nanovlm(model=model)
        second_count = apply_torch_npu_rmsnorm_to_nanovlm(model=model)

    assert first_count == 9
    assert second_count == 0


def test_nanovlm_torch_npu_patch_uses_nested_text_config() -> None:
    model = DummyNanoVLM()
    # Composite Transformers models may expose the outer NanovlmConfig here.
    model.language_model.config = SimpleNamespace(model_type="nanovlm")

    with patch(
        "lmms_engine.models.nanovlm.monkey_patch._load_torch_npu_rms_norm_operator",
        return_value=fake_torch_npu_rms_norm,
    ):
        newly_patched = apply_torch_npu_rmsnorm_to_nanovlm(model=model, strict=True)

    assert newly_patched == 9
    assert all(
        norm._lmms_engine_rms_norm_backend == "torch_npu"
        for norm in get_all_qwen3_norms(model)
    )


def test_nanovlm_rmsnorm_backends_are_mutually_exclusive() -> None:
    model = DummyNanoVLM()

    with patch(
        "lmms_engine.models.nanovlm.monkey_patch._load_liger_rms_norm_patcher",
        return_value=fake_liger_patcher,
    ):
        apply_liger_rmsnorm_to_nanovlm(model=model)

    with patch(
        "lmms_engine.models.nanovlm.monkey_patch._load_torch_npu_rms_norm_operator",
        return_value=fake_torch_npu_rms_norm,
    ):
        try:
            apply_torch_npu_rmsnorm_to_nanovlm(model=model)
        except RuntimeError as exc:
            assert "backend conflict" in str(exc)
        else:
            raise AssertionError("Expected torch_npu RMSNorm to reject a Liger-patched model")
