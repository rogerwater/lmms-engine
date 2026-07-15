from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.models.nanovlm.monkey_patch import (
    apply_torch_npu_rope_to_nanovlm,
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


def fake_torch_npu_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del eps
    output = hidden_states * weight
    reciprocal_rms = torch.ones_like(hidden_states[..., :1])
    return output, reciprocal_rms


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def fake_torch_npu_rotary_mul(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return hidden_states * cos + rotate_half(hidden_states) * sin


def reference_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
    )


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


def test_nanovlm_npu_patches_are_registered_independently() -> None:
    assert "nanovlm" in MONKEY_PATCHER
    assert (
        MONKEY_PATCHER["nanovlm"]["npu_rms_norm"]
        is apply_torch_npu_rmsnorm_to_nanovlm
    )
    assert (
        MONKEY_PATCHER["nanovlm"]["npu_rope"]
        is apply_torch_npu_rope_to_nanovlm
    )
    assert "liger" not in MONKEY_PATCHER["nanovlm"]


def test_nanovlm_torch_npu_rmsnorm_patch_preserves_parameters_and_state_dict() -> None:
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


def test_nanovlm_torch_npu_rmsnorm_patch_is_idempotent() -> None:
    model = DummyNanoVLM()

    with patch(
        "lmms_engine.models.nanovlm.monkey_patch._load_torch_npu_rms_norm_operator",
        return_value=fake_torch_npu_rms_norm,
    ):
        first_count = apply_torch_npu_rmsnorm_to_nanovlm(model=model)
        second_count = apply_torch_npu_rmsnorm_to_nanovlm(model=model)

    assert first_count == 9
    assert second_count == 0


def test_nanovlm_torch_npu_rmsnorm_patch_uses_nested_text_config() -> None:
    model = DummyNanoVLM()
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


def test_nanovlm_torch_npu_rope_matches_forward_and_backward() -> None:
    model = DummyNanoVLM()
    modeling_qwen3 = SimpleNamespace(
        apply_rotary_pos_emb=reference_apply_rotary_pos_emb,
    )
    state_dict_keys_before = tuple(model.state_dict().keys())
    parameter_ids_before = {name: id(parameter) for name, parameter in model.named_parameters()}

    with (
        patch(
            "lmms_engine.models.nanovlm.monkey_patch._load_qwen3_modeling_module",
            return_value=modeling_qwen3,
        ),
        patch(
            "lmms_engine.models.nanovlm.monkey_patch._load_torch_npu_rotary_mul_operator",
            return_value=fake_torch_npu_rotary_mul,
        ),
    ):
        newly_patched = apply_torch_npu_rope_to_nanovlm(model=model, strict=True)
        second_count = apply_torch_npu_rope_to_nanovlm(model=model, strict=True)

    assert newly_patched == 1
    assert second_count == 0
    assert modeling_qwen3._lmms_engine_torch_npu_rope is True
    assert (
        modeling_qwen3._lmms_engine_original_apply_rotary_pos_emb
        is reference_apply_rotary_pos_emb
    )
    assert tuple(model.state_dict().keys()) == state_dict_keys_before
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids_before

    q_reference = torch.randn(2, 4, 5, 8, requires_grad=True)
    k_reference = torch.randn(2, 2, 5, 8, requires_grad=True)
    q_native = q_reference.detach().clone().requires_grad_(True)
    k_native = k_reference.detach().clone().requires_grad_(True)
    cos = torch.randn(2, 5, 8)
    sin = torch.randn(2, 5, 8)

    q_expected, k_expected = reference_apply_rotary_pos_emb(
        q_reference,
        k_reference,
        cos,
        sin,
    )
    q_actual, k_actual = modeling_qwen3.apply_rotary_pos_emb(
        q_native,
        k_native,
        cos,
        sin,
    )
    torch.testing.assert_close(q_actual, q_expected)
    torch.testing.assert_close(k_actual, k_expected)

    q_gradient = torch.randn_like(q_expected)
    k_gradient = torch.randn_like(k_expected)
    torch.autograd.backward((q_expected, k_expected), (q_gradient, k_gradient))
    torch.autograd.backward((q_actual, k_actual), (q_gradient, k_gradient))
    torch.testing.assert_close(q_native.grad, q_reference.grad)
    torch.testing.assert_close(k_native.grad, k_reference.grad)
