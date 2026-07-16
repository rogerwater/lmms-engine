from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.models.nanovlm.monkey_patch import (
    apply_torch_npu_cross_entropy_to_nanovlm,
    apply_torch_npu_rope_to_nanovlm,
    apply_torch_npu_rmsnorm_to_nanovlm,
)
from lmms_engine.models.nanovlm.npu_fused_ops import (
    torch_npu_active_token_causal_lm_loss,
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

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> SimpleNamespace:
        del kwargs
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = F.one_hot(input_ids, num_classes=8).float()
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class DummyQwen3ForCausalLM(nn.Module):
    base_model_prefix = "model"

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3",
            use_return_dict=True,
            vocab_size=13,
        )
        self.model = DummyQwen3Model()
        self.lm_head = nn.Linear(8, self.config.vocab_size, bias=False)
        self.original_forward_calls = 0

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SimpleNamespace:
        self.original_forward_calls += 1
        outputs = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, self.config.vocab_size).float(),
                labels[..., 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


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


def fake_torch_npu_cross_entropy_loss(
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del kwargs
    loss = F.cross_entropy(
        input_tensor,
        target,
        weight=weight,
        reduction=reduction,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    empty = input_tensor.new_empty((0,))
    return loss, input_tensor.new_empty(input_tensor.shape), empty, empty


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
    assert (
        MONKEY_PATCHER["nanovlm"]["npu_cross_entropy"]
        is apply_torch_npu_cross_entropy_to_nanovlm
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
    saved_shapes: list[tuple[int, ...]] = []

    def record_saved_tensor(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record_saved_tensor, lambda tensor: tensor):
        q_actual, k_actual = modeling_qwen3.apply_rotary_pos_emb(
            q_native,
            k_native,
            cos,
            sin,
        )

    torch.testing.assert_close(q_actual, q_expected)
    torch.testing.assert_close(k_actual, k_expected)
    assert tuple(q_native.shape) not in saved_shapes
    assert tuple(k_native.shape) not in saved_shapes

    q_gradient = torch.randn_like(q_expected)
    k_gradient = torch.randn_like(k_expected)
    torch.autograd.backward((q_expected, k_expected), (q_gradient, k_gradient))
    torch.autograd.backward((q_actual, k_actual), (q_gradient, k_gradient))
    torch.testing.assert_close(q_native.grad, q_reference.grad)
    torch.testing.assert_close(k_native.grad, k_reference.grad)


def test_active_token_cross_entropy_matches_full_projection_forward_and_backward() -> None:
    torch.manual_seed(7)
    reference_head = nn.Linear(8, 13, bias=False)
    native_head = nn.Linear(8, 13, bias=False)
    native_head.load_state_dict(reference_head.state_dict())

    reference_hidden = torch.randn(2, 6, 8, requires_grad=True)
    native_hidden = reference_hidden.detach().clone().requires_grad_(True)
    labels = torch.tensor(
        [
            [-100, -100, 3, 4, -100, 5],
            [-100, 2, -100, 6, 7, -100],
        ],
        dtype=torch.long,
    )

    reference_logits = reference_head(reference_hidden[..., :-1, :]).float()
    reference_loss = F.cross_entropy(
        reference_logits.reshape(-1, reference_logits.shape[-1]),
        labels[..., 1:].reshape(-1),
        ignore_index=-100,
    )

    operator_shapes: list[tuple[int, ...]] = []

    def recording_cross_entropy(
        input_tensor: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        operator_shapes.append(tuple(input_tensor.shape))
        return fake_torch_npu_cross_entropy_loss(input_tensor, target, **kwargs)

    native_loss = torch_npu_active_token_causal_lm_loss(
        native_hidden,
        labels,
        native_head,
        recording_cross_entropy,
    )

    torch.testing.assert_close(native_loss, reference_loss)
    assert native_loss.shape == torch.Size([])
    expected_active_tokens = int(labels[..., 1:].ne(-100).sum())
    assert operator_shapes == [(expected_active_tokens, 13)]

    reference_loss.backward()
    native_loss.backward()
    torch.testing.assert_close(native_hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(native_head.weight.grad, reference_head.weight.grad)


def test_active_token_cross_entropy_handles_all_ignored_microbatch() -> None:
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)
    lm_head = nn.Linear(8, 13, bias=False)
    labels = torch.full((2, 4), -100, dtype=torch.long)

    def operator_must_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("native cross entropy must not receive an empty batch")

    loss = torch_npu_active_token_causal_lm_loss(
        hidden_states,
        labels,
        lm_head,
        operator_must_not_run,
    )
    assert loss.item() == 0.0
    assert torch.isfinite(loss)

    loss.backward()
    torch.testing.assert_close(hidden_states.grad, torch.zeros_like(hidden_states))
    torch.testing.assert_close(lm_head.weight.grad, torch.zeros_like(lm_head.weight))


def test_nanovlm_torch_npu_cross_entropy_patch_is_training_only_and_idempotent() -> None:
    model = DummyNanoVLM()
    language_model = model.language_model
    state_dict_keys_before = tuple(model.state_dict().keys())
    parameter_ids_before = {name: id(parameter) for name, parameter in model.named_parameters()}

    with patch(
        "lmms_engine.models.nanovlm.monkey_patch._load_torch_npu_cross_entropy_operator",
        return_value=fake_torch_npu_cross_entropy_loss,
    ):
        first_count = apply_torch_npu_cross_entropy_to_nanovlm(model=model, strict=True)
        second_count = apply_torch_npu_cross_entropy_to_nanovlm(model=model, strict=True)

    assert first_count == 1
    assert second_count == 0
    assert language_model._lmms_engine_torch_npu_cross_entropy is True
    assert language_model._lmms_engine_cross_entropy_backend == "torch_npu"
    assert tuple(model.state_dict().keys()) == state_dict_keys_before
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids_before

    inputs_embeds = torch.randn(2, 5, 8, requires_grad=True)
    labels = torch.tensor(
        [
            [-100, 1, 2, -100, 3],
            [-100, -100, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    language_model.train()
    training_outputs = language_model(inputs_embeds=inputs_embeds, labels=labels)
    assert training_outputs.loss is not None
    assert training_outputs.logits is None
    assert language_model.original_forward_calls == 0
    training_outputs.loss.backward()
    assert inputs_embeds.grad is not None

    language_model.eval()
    eval_outputs = language_model(inputs_embeds=inputs_embeds.detach(), labels=labels)
    assert eval_outputs.loss is not None
    assert eval_outputs.logits.shape == (2, 5, 13)
    assert language_model.original_forward_calls == 1
