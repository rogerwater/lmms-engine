"""Create a NanoVLM initialization checkpoint with a new projector.

The vision tower and language model are reused exactly from the base checkpoint.
Only the multimodal projector is freshly initialized with a deterministic seed.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Union

import torch
from accelerate import init_empty_weights
from transformers import AutoProcessor, AutoTokenizer

from lmms_engine.models.nanovlm import (
    NanovlmForConditionalGeneration,
    build_nanovlm_projector,
)


TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _from_pretrained_with_mistral_regex_fix(auto_class, model_path: str, trust_remote_code: bool):
    kwargs = {"trust_remote_code": trust_remote_code}
    try:
        return auto_class.from_pretrained(model_path, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return auto_class.from_pretrained(model_path, **kwargs)


def _resolve_load_dtype(dtype_name: str) -> Union[str, torch.dtype]:
    if dtype_name == "auto":
        return "auto"
    return TORCH_DTYPES[dtype_name]


def prepare_nanovlm_projector_variant(args: argparse.Namespace) -> None:
    output_path = Path(args.output_path)
    if output_path.exists():
        if not output_path.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_path}")
        if any(output_path.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_path}. "
                "Use a new directory so stale checkpoint shards cannot be mixed into the variant."
            )

    base_model = NanovlmForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=_resolve_load_dtype(args.dtype),
        low_cpu_mem_usage=True,
    )
    config = deepcopy(base_model.config)
    config.projector_type = args.projector_type
    config.projector_hidden_size = args.projector_hidden_size
    config.projector_num_layers = args.projector_num_layers
    config.projector_hidden_act = args.projector_hidden_act
    config.projector_bias = args.projector_bias

    llm_hidden_size = getattr(config.text_config, "hidden_size", None)
    if llm_hidden_size is None:
        raise ValueError("Unable to infer the language model hidden size from the base checkpoint.")

    # Construct the container on meta so the script does not allocate a second
    # random copy of the vision tower and language model.
    with init_empty_weights():
        variant_model = NanovlmForConditionalGeneration(config)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.projector_seed)
        projector = build_nanovlm_projector(
            projector_type=config.projector_type,
            in_dim=config.vision_feature_dim,
            out_dim=llm_hidden_size,
            hidden_dim=config.projector_hidden_size,
            num_layers=config.projector_num_layers,
            act_name=config.projector_hidden_act,
            bias=config.projector_bias,
        )

    base_dtype = next(base_model.parameters()).dtype
    projector.to(dtype=base_dtype)

    # Reusing these module objects guarantees bit-identical backbone weights.
    variant_model.vision_model = base_model.vision_model
    variant_model.language_model = base_model.language_model
    variant_model.multi_modal_projector = projector

    meta_parameters = [name for name, parameter in variant_model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"Variant still contains meta parameters: {meta_parameters[:10]}")

    output_path.mkdir(parents=True, exist_ok=True)
    variant_model.save_pretrained(
        output_path,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )

    tokenizer = _from_pretrained_with_mistral_regex_fix(
        AutoTokenizer,
        args.base_model_path,
        args.trust_remote_code,
    )
    processor = _from_pretrained_with_mistral_regex_fix(
        AutoProcessor,
        args.base_model_path,
        args.trust_remote_code,
    )
    tokenizer.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    projector_parameter_count = sum(parameter.numel() for parameter in projector.parameters())
    summary = {
        "base_model_path": args.base_model_path,
        "output_path": str(output_path),
        "projector_type": config.projector_type,
        "projector_hidden_size": config.projector_hidden_size,
        "projector_num_layers": config.projector_num_layers,
        "projector_hidden_act": config.projector_hidden_act,
        "projector_bias": config.projector_bias,
        "projector_seed": args.projector_seed,
        "projector_parameter_count": projector_parameter_count,
        "dtype": str(base_dtype),
    }
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a NanoVLM initialization checkpoint that reuses the exact base "
            "vision/language weights and freshly initializes one projector variant."
        )
    )
    parser.add_argument("--base-model-path", required=True, help="Base NanoVLM_Init checkpoint.")
    parser.add_argument("--output-path", required=True, help="New, empty output checkpoint directory.")
    parser.add_argument(
        "--projector-type",
        choices=["linear", "mlp", "swiglu"],
        default="mlp",
    )
    parser.add_argument(
        "--projector-hidden-size",
        type=int,
        default=None,
        help="Defaults to the language hidden size. For parameter-matched SwiGLU, use 672.",
    )
    parser.add_argument(
        "--projector-num-layers",
        type=int,
        default=2,
        help="Number of Linear layers for an MLP projector.",
    )
    parser.add_argument(
        "--projector-hidden-act",
        default="gelu",
        help="Activation between MLP Linear layers. SwiGLU always uses SiLU gating.",
    )
    parser.add_argument(
        "--projector-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--projector-seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["auto", *TORCH_DTYPES],
        default="auto",
        help="Loading dtype for the base checkpoint; the new projector uses the same dtype.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


if __name__ == "__main__":
    prepare_nanovlm_projector_variant(parse_args())
