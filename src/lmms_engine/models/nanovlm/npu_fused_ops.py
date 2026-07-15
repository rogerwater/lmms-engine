"""Pure forward implementations used by NanoVLM torch_npu patches."""

from __future__ import annotations

from typing import Any, Callable


def make_torch_npu_apply_rotary_pos_emb(
    rotary_mul: Callable[..., Any],
) -> Callable[..., tuple[Any, Any]]:
    """Build a Qwen3-compatible RoPE function backed by npu_rotary_mul."""

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
        q_embed = rotary_mul(q, cos, sin)
        k_embed = rotary_mul(k, cos, sin)
        return q_embed, k_embed

    return apply_rotary_pos_emb


__all__ = ["make_torch_npu_apply_rotary_pos_emb"]
