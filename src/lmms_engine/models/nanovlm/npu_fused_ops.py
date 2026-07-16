"""Fused implementations used by NanoVLM torch_npu patches."""

from __future__ import annotations

from typing import Any, Callable

import torch


def _make_memory_efficient_rotary_mul(
    rotary_mul: Callable[..., Any],
) -> Callable[..., Any]:
    """Use RotaryMul in backward without retaining the large q/k input.

    torch_npu's automatic backward may retain the RotaryMul input so that it
    can also produce gradients for cos and sin. Qwen3 creates cos and sin under
    no_grad, so its training path only needs the input gradient. This custom
    autograd function retains cos and sin and evaluates the transposed rotary
    transform directly in backward.
    """

    class MemoryEfficientRotaryMul(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, hidden_states: Any, cos: Any, sin: Any) -> Any:
            ctx.save_for_backward(cos, sin)
            return rotary_mul(hidden_states, cos, sin)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None, None]:
            if grad_output is None:
                return None, None, None

            cos, sin = ctx.saved_tensors
            if sin.shape[-1] % 2 != 0:
                raise RuntimeError(
                    "npu_rope requires an even rotary head dimension, "
                    f"but received {sin.shape[-1]}."
                )

            # For y = x * cos + rotate_half(x) * sin, the exact input
            # gradient is another RotaryMul with the two sin halves swapped
            # and negated. The temporary has no attention-head dimension and
            # is therefore much smaller than retaining q/k.
            sin_first, sin_second = sin.chunk(2, dim=-1)
            transposed_sin = torch.cat((sin_second, sin_first), dim=-1).neg_()
            grad_hidden_states = rotary_mul(
                grad_output,
                cos,
                transposed_sin,
            )
            return grad_hidden_states, None, None

    def memory_efficient_rotary_mul(
        hidden_states: Any,
        cos: Any,
        sin: Any,
    ) -> Any:
        # Preserve full autograd semantics for non-Qwen callers that request
        # gradients for rotary coefficients. Qwen3's cos/sin do not require
        # gradients and therefore take the memory-efficient path.
        if (
            torch.is_grad_enabled()
            and hidden_states.requires_grad
            and not cos.requires_grad
            and not sin.requires_grad
        ):
            return MemoryEfficientRotaryMul.apply(hidden_states, cos, sin)
        return rotary_mul(hidden_states, cos, sin)

    return memory_efficient_rotary_mul


def make_torch_npu_apply_rotary_pos_emb(
    rotary_mul: Callable[..., Any],
) -> Callable[..., tuple[Any, Any]]:
    """Build a Qwen3-compatible RoPE function backed by npu_rotary_mul."""

    memory_efficient_rotary_mul = _make_memory_efficient_rotary_mul(rotary_mul)

    def apply_rotary_pos_emb(
        q: Any,
        k: Any,
        cos: Any,
        sin: Any,
        position_ids: Any = None,
        unsqueeze_dim: int = 1,
    ) -> tuple[Any, Any]:
        del position_ids
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        # The default rotary_mode is "half", matching Qwen3 rotate_half.
        q_embed = memory_efficient_rotary_mul(q, cos, sin)
        k_embed = memory_efficient_rotary_mul(k, cos, sin)
        return q_embed, k_embed

    return apply_rotary_pos_emb


def torch_npu_active_token_causal_lm_loss(
    hidden_states: Any,
    labels: Any,
    lm_head: Any,
    cross_entropy_loss: Callable[..., Any],
    *,
    ignore_index: int = -100,
    num_items_in_batch: Any = None,
) -> Any:
    """Compute causal-LM loss only for non-ignored target tokens.

    Qwen3's default loss first materializes logits for every sequence
    position, even though multimodal instruction data commonly masks most of
    those positions with ``ignore_index``. Selecting active hidden states
    before ``lm_head`` avoids both the vocabulary projection and the fused
    cross-entropy workspace for ignored positions.
    """

    if hidden_states.ndim < 2:
        raise ValueError(
            "npu_cross_entropy expects hidden_states with a sequence dimension, "
            f"but received shape={tuple(hidden_states.shape)}."
        )
    if tuple(labels.shape) != tuple(hidden_states.shape[:-1]):
        raise ValueError(
            "npu_cross_entropy expects labels to match hidden_states without "
            f"the hidden dimension, but received labels={tuple(labels.shape)} "
            f"and hidden_states={tuple(hidden_states.shape)}."
        )
    if hidden_states.shape[-2] < 2:
        raise ValueError(
            "npu_cross_entropy requires a sequence length of at least two for "
            "causal label shifting."
        )

    hidden_size = hidden_states.shape[-1]
    shifted_hidden_states = hidden_states[..., :-1, :].reshape(-1, hidden_size)
    shifted_labels = labels[..., 1:].reshape(-1).to(
        device=hidden_states.device,
        dtype=torch.long,
    )
    active_mask = shifted_labels.ne(ignore_index)
    active_hidden_states = shifted_hidden_states[active_mask]
    active_labels = shifted_labels[active_mask]

    # aclnnCrossEntropyLoss requires N > 0. Returning a differentiable zero is
    # also safer than the NaN produced by mean reduction when every label in a
    # microbatch is ignored.
    if active_labels.numel() == 0:
        zero_loss = hidden_states.float().sum() * 0.0
        for parameter in lm_head.parameters():
            zero_loss = zero_loss + parameter.float().sum() * 0.0
        return zero_loss

    # Match Transformers' ForCausalLMLoss numerical policy: cross entropy is
    # evaluated with FP32 logits even when the model itself uses BF16/FP16.
    active_logits = lm_head(active_hidden_states).float()
    reduction = "sum" if num_items_in_batch is not None else "mean"
    result = cross_entropy_loss(
        active_logits,
        active_labels,
        reduction=reduction,
    )
    if isinstance(result, (tuple, list)):
        if not result:
            raise RuntimeError("torch_npu.npu_cross_entropy_loss returned an empty result.")
        loss = result[0]
    else:
        loss = result

    # The native operator returns shape [1] for mean/sum, while Trainer expects
    # the scalar shape used by torch.nn.functional.cross_entropy.
    if loss.numel() == 1:
        loss = loss.reshape(())

    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(device=loss.device)
        loss = loss / num_items_in_batch
    return loss


__all__ = [
    "make_torch_npu_apply_rotary_pos_emb",
    "torch_npu_active_token_causal_lm_loss",
]
