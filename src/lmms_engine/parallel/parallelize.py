from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmms_engine.train.config import TrainingArguments

MODEL_TO_PARALLEL_METHOD = {}
_UNAVAILABLE_PARALLEL_METHODS = {}


def _register_parallel_method(model_types, module_name: str, function_name: str):
    if isinstance(model_types, str):
        model_types = (model_types,)

    try:
        module = import_module(module_name, package=__package__)
        parallelize_fn = getattr(module, function_name)
    except ImportError as exc:
        for model_type in model_types:
            _UNAVAILABLE_PARALLEL_METHODS[model_type] = exc
        return

    for model_type in model_types:
        MODEL_TO_PARALLEL_METHOD[model_type] = parallelize_fn


_register_parallel_method("qwen3_moe", ".qwen3_moe.parallelize", "apply_qwen3_moe_parallelize_fn")
_register_parallel_method("qwen3_5_moe", ".qwen3_5_moe.parallelize", "apply_qwen3_5_moe_parallelize_fn")
_register_parallel_method(
    ("qwen3_omni_moe", "qwen3_omni_moe_thinker"),
    ".qwen3_omni_moe.parallelize",
    "apply_qwen3_omni_moe_parallelize_fn",
)
_register_parallel_method("qwen3_vl", ".qwen3_vl.parallelize", "apply_qwen3_vl_parallelize_fn")
_register_parallel_method("qwen3_vl_moe", ".qwen3_vl_moe.parallelize", "apply_qwen3_vl_moe_parallelize_fn")


def apply_parallelize(model, model_type, train_args: "TrainingArguments", **kwargs):
    """
    Apply parallelization based on model type.

    Args:
        model: The model to parallelize
        model_type: Key in MODEL_TO_PARALLEL_METHOD (e.g., "qwen3_moe")
        train_args: Training configuration

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type in _UNAVAILABLE_PARALLEL_METHODS:
        raise ImportError(
            f"Model type {model_type} requires an optional parallel backend that is unavailable. "
            f"Original import error: {_UNAVAILABLE_PARALLEL_METHODS[model_type]}"
        ) from _UNAVAILABLE_PARALLEL_METHODS[model_type]

    if model_type not in MODEL_TO_PARALLEL_METHOD:
        raise ValueError(f"Model type {model_type} not supported")

    return MODEL_TO_PARALLEL_METHOD[model_type](model, train_args, **kwargs)
