"""NanoVLM-specific monkey patches.

NanoVLM is registered as ``model_type=nanovlm`` while its text backbone is a
Qwen3 causal language model.  The generic Qwen3 Liger registration therefore
does not match the top-level model.  This module provides a deliberately narrow
instance patch that only replaces the RMSNorm forwards inside the text model.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from loguru import logger

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


def _load_liger_rms_norm_patcher() -> Callable[..., None]:
    try:
        from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module
    except ImportError as exc:
        raise ImportError(
            "NanoVLM Liger RMSNorm was requested, but liger-kernel could not "
            "be imported. Install a Liger/Triton version compatible with the "
            "current accelerator environment."
        ) from exc
    return _patch_rms_norm_module


def _get_qwen3_base_model(model: Any, strict: bool) -> tuple[Any, Any]:
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise ValueError("NanoVLM Liger patch requires model.language_model.")

    language_config = getattr(language_model, "config", None)
    language_model_type = getattr(language_config, "model_type", None)
    if language_model_type != "qwen3":
        message = (
            "NanoVLM Liger RMSNorm currently supports only a Qwen3 text "
            f"backbone, but found model_type={language_model_type!r}."
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)

    base_model_prefix = getattr(language_model, "base_model_prefix", "model")
    base_model = getattr(language_model, base_model_prefix, None)
    if base_model is None:
        base_model = getattr(language_model, "model", None)
    if base_model is None:
        raise ValueError(
            "Unable to locate the Qwen3 base model under NanoVLM's "
            f"language_model (base_model_prefix={base_model_prefix!r})."
        )

    return language_model, base_model


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

    if not modules:
        raise ValueError("No Qwen3 RMSNorm modules were found under NanoVLM.language_model.")
    return modules


def _patch_rms_norm_module(
    module: Any,
    patcher: Callable[..., None],
    in_place: bool,
    strict: bool,
) -> bool:
    if not hasattr(module, "weight"):
        raise TypeError(f"Expected an RMSNorm-like module with a weight parameter, got {type(module)!r}.")

    state_dict_keys_before = tuple(module.state_dict().keys())
    already_patched = bool(getattr(module, "_lmms_engine_liger_rms_norm", False))

    patcher_signature = inspect.signature(patcher)
    patcher_kwargs: dict[str, Any] = {}
    if "offset" in patcher_signature.parameters:
        patcher_kwargs["offset"] = 0.0
    if "casting_mode" in patcher_signature.parameters:
        patcher_kwargs["casting_mode"] = "llama"
    if "in_place" in patcher_signature.parameters:
        patcher_kwargs["in_place"] = in_place

    patcher(module, **patcher_kwargs)

    # Older Liger helpers may not expose in_place as a keyword, while
    # LigerRMSNorm.forward still reads the value from the module instance.
    module.in_place = in_place
    module._lmms_engine_liger_rms_norm = True

    state_dict_keys_after = tuple(module.state_dict().keys())
    if strict and state_dict_keys_after != state_dict_keys_before:
        raise RuntimeError(
            "Patching RMSNorm changed its state-dict keys: "
            f"before={state_dict_keys_before}, after={state_dict_keys_after}."
        )

    return not already_patched


@MONKEY_PATCHER.register("nanovlm", "liger")
def apply_liger_rmsnorm_to_nanovlm(
    model: Any = None,
    rms_norm: bool = True,
    rms_norm_in_place: bool = True,
    rope: bool = False,
    swiglu: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = False,
    strict: bool = True,
    use_rmpad: bool = False,
) -> int:
    """Apply only Liger RMSNorm to NanoVLM's internal Qwen3 model.

    Other Liger kernels are intentionally rejected for now so that enabling
    ``trainer_args.use_liger_kernel`` has a single, attributable effect.
    ``use_rmpad`` is accepted because the runner always forwards it, but it
    does not alter the RMSNorm patch.

    Returns:
        Number of RMSNorm modules newly patched during this call.
    """

    del use_rmpad

    if model is None:
        raise ValueError("A constructed NanoVLM model instance is required for the Liger patch.")

    unsupported = {
        "rope": rope,
        "swiglu": swiglu,
        "cross_entropy": cross_entropy,
        "fused_linear_cross_entropy": fused_linear_cross_entropy,
    }
    requested_unsupported = [name for name, enabled in unsupported.items() if enabled]
    if requested_unsupported:
        raise ValueError(
            "The NanoVLM Liger patch currently supports RMSNorm only. "
            f"Disable: {', '.join(requested_unsupported)}."
        )

    if not rms_norm:
        logger.info("NanoVLM Liger RMSNorm is disabled; no modules were patched.")
        return 0

    language_model, base_model = _get_qwen3_base_model(model, strict=strict)
    rms_norm_modules = _collect_qwen3_rms_norm_modules(base_model, strict=strict)
    patcher = _load_liger_rms_norm_patcher()

    newly_patched = 0
    for module_name, module in rms_norm_modules:
        try:
            newly_patched += int(
                _patch_rms_norm_module(
                    module,
                    patcher=patcher,
                    in_place=rms_norm_in_place,
                    strict=strict,
                )
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to apply Liger RMSNorm to {module_name}: {exc}") from exc

    expected_count = 2 * len(base_model.layers) + 1
    matched_count = len(rms_norm_modules)
    if strict and matched_count != expected_count:
        raise RuntimeError(
            "Unexpected Qwen3 RMSNorm count: "
            f"matched={matched_count}, expected={expected_count}."
        )

    logger.info(
        "Applied NanoVLM Liger RMSNorm patch: "
        f"text_model_type={language_model.config.model_type}, "
        f"decoder_layers={len(base_model.layers)}, matched={matched_count}, "
        f"newly_patched={newly_patched}, in_place={rms_norm_in_place}"
    )
    return newly_patched


__all__ = ["apply_liger_rmsnorm_to_nanovlm"]
