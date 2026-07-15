"""Smoke-test Liger RMSNorm forward and backward on an Ascend NPU.

This script compares LigerRMSNorm against a small PyTorch reference
implementation. The first invocation can be slow because Triton compiles and
caches the kernel.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version

import torch
import torch_npu  # noqa: F401  # Registers the NPU backend with PyTorch.

from liger_kernel.transformers import LigerRMSNorm


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def rms_norm_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states_fp32 = hidden_states.float()
    variance = hidden_states_fp32.square().mean(dim=-1, keepdim=True)
    normalized = hidden_states_fp32 * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(input_dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="NPU device index")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument(
        "--in-place",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse the upstream-gradient storage for the RMSNorm input "
            "gradient during backward (default: disabled)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is False; no usable NPU was found")

    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    torch.npu.set_device(args.device)

    device = torch.device(f"npu:{args.device}")
    dtype = DTYPES[args.dtype]
    shape = (args.batch_size, args.sequence_length, args.hidden_size)

    print("Environment")
    print(f"  torch:          {torch.__version__}")
    print(f"  torch_npu:      {package_version('torch-npu')}")
    print(f"  triton-ascend:  {package_version('triton-ascend')}")
    print(f"  liger-kernel:   {package_version('liger-kernel')}")
    print(f"  device:         {device}")
    print(f"  dtype:          {dtype}")
    print(f"  input shape:    {shape}")
    print(f"  in_place:       {args.in_place}")

    liger_norm = LigerRMSNorm(
        hidden_size=args.hidden_size,
        eps=args.eps,
        in_place=args.in_place,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        liger_norm.weight.uniform_(0.8, 1.2)

    base_input = torch.randn(shape, device=device, dtype=dtype)
    liger_input = base_input.detach().clone().requires_grad_(True)
    reference_input = base_input.detach().clone().requires_grad_(True)
    reference_weight = liger_norm.weight.detach().clone().requires_grad_(True)

    liger_output = liger_norm(liger_input)
    reference_output = rms_norm_reference(
        reference_input,
        reference_weight,
        args.eps,
    )

    grad_output = torch.randn_like(liger_output)

    # Liger RMSNorm with in_place=True may reuse dY's storage for dX during
    # backward. Give each implementation its own upstream-gradient tensor so
    # that the Liger backward cannot change the gradient subsequently consumed
    # by the PyTorch reference backward.
    liger_grad_output = grad_output.detach().clone()
    reference_grad_output = grad_output.detach().clone()

    reference_output.backward(reference_grad_output)
    liger_output.backward(liger_grad_output)
    torch.npu.synchronize()

    tensors = {
        "Liger output": liger_output,
        "Liger input gradient": liger_input.grad,
        "Liger weight gradient": liger_norm.weight.grad,
    }
    for name, tensor in tensors.items():
        if tensor is None:
            raise RuntimeError(f"{name} was not produced")
        if not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"{name} contains NaN or Inf")

    comparisons = {
        "forward output": (liger_output, reference_output),
        "input gradient": (liger_input.grad, reference_input.grad),
        "weight gradient": (liger_norm.weight.grad, reference_weight.grad),
    }

    print("\nNumerical comparison")
    for name, (actual, expected) in comparisons.items():
        actual_fp32 = actual.detach().float()
        expected_fp32 = expected.detach().float()
        difference = (actual_fp32 - expected_fp32).abs()
        print(
            f"  {name:16s} "
            f"max_abs_diff={difference.max().item():.6e}, "
            f"mean_abs_diff={difference.mean().item():.6e}"
        )
        torch.testing.assert_close(
            actual_fp32,
            expected_fp32,
            rtol=args.rtol,
            atol=args.atol,
            msg=lambda msg, name=name: f"{name} mismatch:\n{msg}",
        )

    print(
        "\nPASS: Liger RMSNorm NPU forward/backward matches the PyTorch "
        f"reference (in_place={args.in_place})."
    )


if __name__ == "__main__":
    main()
