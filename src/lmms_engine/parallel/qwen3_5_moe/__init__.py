try:
    from .parallelize import apply_qwen3_5_moe_parallel, apply_qwen3_5_moe_parallelize_fn
except ImportError as exc:
    _PARALLEL_IMPORT_ERROR = exc

    def apply_qwen3_5_moe_parallel(*args, **kwargs):
        raise ImportError(
            "Qwen3.5-MoE parallelization requires a transformers build that provides "
            "`transformers.models.qwen3_5_moe`."
        ) from _PARALLEL_IMPORT_ERROR

    def apply_qwen3_5_moe_parallelize_fn(*args, **kwargs):
        raise ImportError(
            "Qwen3.5-MoE parallelization requires a transformers build that provides "
            "`transformers.models.qwen3_5_moe`."
        ) from _PARALLEL_IMPORT_ERROR

try:
    from .style import Qwen3_5MoeParallelStyle
except ImportError as exc:
    _STYLE_IMPORT_ERROR = exc

    class Qwen3_5MoeParallelStyle:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Qwen3.5-MoE parallel style requires a transformers build that provides "
                "`transformers.models.qwen3_5_moe`."
            ) from _STYLE_IMPORT_ERROR

__all__ = [
    "apply_qwen3_5_moe_parallel",
    "apply_qwen3_5_moe_parallelize_fn",
    "Qwen3_5MoeParallelStyle",
]
