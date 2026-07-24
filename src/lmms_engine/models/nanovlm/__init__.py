from lmms_engine.mapping_func import register_model

from .configuration_nanovlm import NanovlmConfig
from .monkey_patch import (
    apply_torch_npu_cross_entropy_to_nanovlm,
    apply_torch_npu_rope_to_nanovlm,
    apply_torch_npu_rmsnorm_to_nanovlm,
)
from .modeling_nanovlm import (
    NanovlmForConditionalGeneration,
    NanovlmSwiGLUProjector,
    build_nanovlm_projector,
)

register_model(
    "nanovlm",
    NanovlmConfig,
    NanovlmForConditionalGeneration,
    model_general_type="image_text_to_text",
)

__all__ = [
    "NanovlmConfig",
    "NanovlmForConditionalGeneration",
    "NanovlmSwiGLUProjector",
    "build_nanovlm_projector",
    "apply_torch_npu_cross_entropy_to_nanovlm",
    "apply_torch_npu_rope_to_nanovlm",
    "apply_torch_npu_rmsnorm_to_nanovlm",
]
