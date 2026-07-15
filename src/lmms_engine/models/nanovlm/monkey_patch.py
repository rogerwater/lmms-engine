"""Independent torch_npu monkey patches for NanoVLM's Qwen3 backbone."""

from __future__ import annotations

from types import MethodType
from typing import Any, Callable

from loguru import logger

from lmms_engine.models.monkey_patch import MONKEY_PATCHER

from .npu_fused_ops import make_torch_npu_apply_rotary_pos_emb


def _load_torch_npu_operator(name: str) -> Callable[..., Any]:
    try:
        import torch_npu
    except ImportError as exc:
        raise ImportError(
            f"NanoVLM {name} was requested, but torch_npu could not be imported. "
            "Install torch_npu for the current PyTorch and CANN versions."
        ) from exc

    operator = getattr(torch_npu, name, None)
    if operator is None:
        raise RuntimeError(f"The installed torch_npu does not expose torch_npu.{name}.")
    return operator


def _load_torch_npu_rms_norm_operator() -> Callable[..., Any]:
    return _load_torch_npu_operator("npu_rms_norm")


def _load_torch_npu_rotary_mul_operator() -> Callable[..., Any]:
    return _load_torch_npu_operator("npu_rotary_mul")


def _load_qwen3_modeling_module() -> Any:
    from transformers.models.qwen3 import modeling_qwen3

    return modeling_qwen3


def _config_model_type(config: Any) -> Any:
    if isinstance(config, dict):
        return config.get("model_type")
    return getattr(config, "model_type", None)


def _get_qwen3_base_model(model: Any, strict: bool) -> tuple[Any, str]:
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise ValueError("NanoVLM NPU patches require model.language_model.")

    base_model_prefix = getattr(language_model, "base_model_prefix", "model")
    base_model = getattr(language_model, base_model_prefix, None)
    if base_model is None:
        base_model = getattr(language_model, "model", None)
    if base_model is None:
        raise ValueError(
            "Unable to locate the Qwen3 base model under NanoVLM's "
            f"language_model (base_model_prefix={base_model_prefix!r})."
        )

    # Transformers composite models can expose the outer NanovlmConfig through
    # language_model.config. The nested text_config is authoritative here.
    top_level_config = getattr(model, "config", None)
    text_config = getattr(top_level_config, "text_config", None)
    model_types = {
        "model.config.text_config": _config_model_type(text_config),
        "language_model.config": _config_model_type(getattr(language_model, "config", None)),
        "base_model.config": _config_model_type(getattr(base_model, "config", None)),
    }
    qwen3_class_names = {"Qwen3ForCausalLM", "Qwen3Model"}
    class_names = {
        "language_model": type(language_model).__name__,
        "base_model": type(base_model).__name__,
    }
    is_qwen3 = "qwen3" in model_types.values() or any(
        class_name in qwen3_class_names for class_name in class_names.values()
    )
    if not is_qwen3:
        message = (
            "NanoVLM NPU patches currently support only a Qwen3 text backbone, "
            f"but detected model_types={model_types!r}, class_names={class_names!r}."
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)

    detected_model_type = "qwen3" if is_qwen3 else str(
        next((value for value in model_types.values() if value is not None), "unknown")
    )
    return base_model, detected_model_type


def _collect_qwen3_rms_norm_modules(base_model: Any, strict: bool) -> list[tuple[str, Any]]:
    modules: list[tuple[str, Any]] = []

    final_norm = getattr(base_model, "norm", None)
    if final_norm is None:
        if strict:
            raise ValueError("Qwen3 base model does not expose the final norm as model.norm.")
        logger.warning("Skipping missing Qwen3 final norm: model.norm")
    else:
        modules.append(("model.norm", final_norm))

    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise ValueError("Qwen3 base model does not expose decoder layers as model.layers.")

    for layer_index, decoder_layer in enumerate(layers):
        for attribute in ("input_layernorm", "post_attention_layernorm"):
            module = getattr(decoder_layer, attribute, None)
            module_name = f"model.layers.{layer_index}.{attribute}"
            if module is None:
                if strict:
                    raise ValueError(f"Missing expected Qwen3 RMSNorm module: {module_name}")
                logger.warning(f"Skipping missing Qwen3 RMSNorm module: {module_name}")
                continue
            modules.append((module_name, module))

        self_attention = getattr(decoder_layer, "self_attn", None)
        if self_attention is None:
            if strict:
                raise ValueError(
                    f"Missing expected Qwen3 self-attention module: model.layers.{layer_index}.self_attn"
                )
            logger.warning(
                f"Skipping missing Qwen3 self-attention module: model.layers.{layer_index}.self_attn"
            )
            continue

        for attribute in ("q_norm", "k_norm"):
            module = getattr(self_attention, attribute, None)
            module_name = f"model.layers.{layer_index}.self_attn.{attribute}"
            if module is None:
                if strict:
                    raise ValueError(f"Missing expected Qwen3 RMSNorm module: {module_name}")
                logger.warning(f"Skipping missing Qwen3 RMSNorm module: {module_name}")
                continue
            modules.append((module_name, module))

    if not modules:
        raise ValueError("No Qwen3 RMSNorm modules were found under NanoVLM.language_model.")
    return modules


def _rms_norm_epsilon_attribute(module: Any) -> str:
    for attribute in ("variance_epsilon", "eps"):
        if hasattr(module, attribute):
            return attribute
    raise TypeError(
        "Expected an RMSNorm-like module with a variance_epsilon or eps attribute, "
        f"got {type(module)!r}."
    )


def _make_torch_npu_rms_norm_forward(
    operator: Callable[..., Any],
    epsilon_attribute: str,
) -> Callable[..., Any]:
    def torch_npu_rms_norm_forward(module: Any, hidden_states: Any) -> Any:
        result = operator(hidden_states, module.weight, getattr(module, epsilon_attribute))
        if isinstance(result, (tuple, list)):
            if not result:
                raise RuntimeError("torch_npu.npu_rms_norm returned an empty result.")
            return result[0]
        return result

    return torch_npu_rms_norm_forward


def _patch_torch_npu_rms_norm_module(
    module: Any,
    operator: Callable[..., Any],
) -> bool:
    backend = getattr(module, "_lmms_engine_rms_norm_backend", None)
    if backend not in (None, "torch_npu"):
        raise RuntimeError(
            f"RMSNorm is already patched with backend={backend!r}; "
            "only one RMSNorm backend may be enabled."
        )
    if getattr(module, "_lmms_engine_torch_npu_rms_norm", False):
        return False
    if not hasattr(module, "weight"):
        raise TypeError(f"Expected an RMSNorm-like module with a weight, got {type(module)!r}.")

    epsilon_attribute = _rms_norm_epsilon_attribute(module)
    state_dict_keys_before = tuple(module.state_dict().keys())
    weight_id_before = id(module.weight)

    native_forward = _make_torch_npu_rms_norm_forward(operator, epsilon_attribute)
    module.forward = MethodType(native_forward, module)
    module._lmms_engine_torch_npu_rms_norm = True
    module._lmms_engine_rms_norm_backend = "torch_npu"

    if tuple(module.state_dict().keys()) != state_dict_keys_before:
        raise RuntimeError("Patching RMSNorm changed its state-dict keys.")
    if id(module.weight) != weight_id_before:
        raise RuntimeError("Patching RMSNorm replaced its weight parameter object.")
    return True


@MONKEY_PATCHER.register("nanovlm", "npu_rms_norm")
def apply_torch_npu_rmsnorm_to_nanovlm(
    model: Any = None,
    strict: bool = True,
    use_rmpad: bool = False,
) -> int:
    """Patch every Qwen3 RMSNorm instance with torch_npu.npu_rms_norm."""

    del use_rmpad
    if model is None:
        raise ValueError("A constructed NanoVLM model is required for npu_rms_norm.")

    base_model, text_model_type = _get_qwen3_base_model(model, strict=strict)
    rms_norm_modules = _collect_qwen3_rms_norm_modules(base_model, strict=strict)
    expected_count = 4 * len(base_model.layers) + 1
    matched_count = len(rms_norm_modules)
    if strict and matched_count != expected_count:
        raise RuntimeError(
            f"Unexpected Qwen3 RMSNorm count: matched={matched_count}, expected={expected_count}."
        )

    # Validate the full set before mutating any module.
    for module_name, module in rms_norm_modules:
        try:
            backend = getattr(module, "_lmms_engine_rms_norm_backend", None)
            if backend not in (None, "torch_npu"):
                raise RuntimeError(f"RMSNorm backend is already {backend!r}.")
            if not hasattr(module, "weight"):
                raise TypeError(f"Expected a weight parameter, got {type(module)!r}.")
            _rms_norm_epsilon_attribute(module)
        except Exception as exc:
            raise RuntimeError(f"Cannot apply npu_rms_norm to {module_name}: {exc}") from exc

    operator = _load_torch_npu_rms_norm_operator()
    newly_patched = 0
    for module_name, module in rms_norm_modules:
        try:
            newly_patched += int(_patch_torch_npu_rms_norm_module(module, operator))
        except Exception as exc:
            raise RuntimeError(f"Failed to apply npu_rms_norm to {module_name}: {exc}") from exc

    logger.info(
        "Applied NanoVLM npu_rms_norm patch: "
        f"text_model_type={text_model_type}, decoder_layers={len(base_model.layers)}, "
        f"matched={matched_count}, newly_patched={newly_patched}"
    )
    return newly_patched


@MONKEY_PATCHER.register("nanovlm", "npu_rope")
def apply_torch_npu_rope_to_nanovlm(
    model: Any = None,
    strict: bool = True,
    use_rmpad: bool = False,
) -> int:
    """Replace Qwen3 RoPE with two torch_npu.npu_rotary_mul calls."""

    del use_rmpad
    if model is None:
        raise ValueError("A constructed NanoVLM model is required for npu_rope.")

    base_model, text_model_type = _get_qwen3_base_model(model, strict=strict)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise ValueError("Qwen3 base model does not expose decoder layers as model.layers.")

    modeling_qwen3 = _load_qwen3_modeling_module()
    current_forward = getattr(modeling_qwen3, "apply_rotary_pos_emb", None)
    if not callable(current_forward):
        raise RuntimeError("Transformers Qwen3 does not expose apply_rotary_pos_emb.")

    if getattr(modeling_qwen3, "_lmms_engine_torch_npu_rope", False):
        logger.info("NanoVLM npu_rope is already applied; no function was patched.")
        return 0

    operator = _load_torch_npu_rotary_mul_operator()
    native_forward = make_torch_npu_apply_rotary_pos_emb(operator)
    modeling_qwen3._lmms_engine_original_apply_rotary_pos_emb = current_forward
    modeling_qwen3.apply_rotary_pos_emb = native_forward
    modeling_qwen3._lmms_engine_torch_npu_rope = True

    logger.info(
        "Applied NanoVLM npu_rope patch: "
        f"text_model_type={text_model_type}, decoder_layers={len(layers)}, "
        "patched_functions=1"
    )
    return 1


__all__ = [
    "apply_torch_npu_rmsnorm_to_nanovlm",
    "apply_torch_npu_rope_to_nanovlm",
]
