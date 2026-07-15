"""Benchmark four RMSNorm implementations on an Ascend NPU.

Providers:
  1. A Qwen3-compatible PyTorch reference.
  2. Liger RMSNorm with an out-of-place backward.
  3. Liger RMSNorm with an in-place backward.
  4. torch_npu.npu_rms_norm.

The script validates forward and backward numerics before measuring latency.
Triton compilation is kept in the warm-up phase and is not included in the
steady-state timing.  The forward-backward benchmark gives every provider its
own cloned upstream gradient because Liger's in-place backward may reuse that
gradient's storage for the input gradient.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

import torch
import torch.nn as nn
import torch_npu
from liger_kernel.transformers import LigerRMSNorm


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class Qwen3ReferenceRMSNorm(nn.Module):
    """PyTorch RMSNorm with the casting order used by Hugging Face Qwen3."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.float()
        variance = hidden_states_fp32.square().mean(dim=-1, keepdim=True)
        normalized = hidden_states_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


class NpuRMSNorm(nn.Module):
    """Thin module wrapper around torch_npu.npu_rms_norm."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)
        if isinstance(result, (tuple, list)):
            return result[0]
        return result


@dataclass
class Provider:
    name: str
    module: nn.Module


@dataclass
class BenchmarkResult:
    provider: str
    mode: str
    iteration_ms: float
    samples_ms: list[float]
    peak_memory_mib: float


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="NPU device index")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward-backward", "all"),
        default="all",
    )
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument(
        "--check-correctness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check outputs and gradients before benchmarking (default: enabled)",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    return args


def ensure_runtime_support() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is False; no usable NPU was found")
    if not hasattr(torch_npu, "npu_rms_norm"):
        raise RuntimeError(
            "The installed torch_npu does not expose torch_npu.npu_rms_norm; "
            "the requested four-provider benchmark cannot run."
        )


def make_providers(
    hidden_size: int,
    eps: float,
    device: torch.device,
    dtype: torch.dtype,
    initial_weight: torch.Tensor,
) -> list[Provider]:
    providers = [
        Provider("pytorch_qwen3", Qwen3ReferenceRMSNorm(hidden_size, eps)),
        Provider(
            "liger_out_of_place",
            LigerRMSNorm(hidden_size=hidden_size, eps=eps, in_place=False),
        ),
        Provider(
            "liger_in_place",
            LigerRMSNorm(hidden_size=hidden_size, eps=eps, in_place=True),
        ),
        Provider("torch_npu_native", NpuRMSNorm(hidden_size, eps)),
    ]

    for provider in providers:
        provider.module.to(device=device, dtype=dtype)
        with torch.no_grad():
            provider.module.weight.copy_(initial_weight)
    return providers


def difference_summary(actual: torch.Tensor, expected: torch.Tensor) -> str:
    difference = (actual.detach().float() - expected.detach().float()).abs()
    return (
        f"max_abs_diff={difference.max().item():.6e}, "
        f"mean_abs_diff={difference.mean().item():.6e}"
    )


def assert_tensor_close(
    name: str,
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
    rtol: float,
    atol: float,
) -> None:
    if actual is None or expected is None:
        raise AssertionError(f"{name}: a required tensor or gradient was not produced")
    if not torch.isfinite(actual).all().item():
        raise AssertionError(f"{name}: actual tensor contains NaN or Inf")
    if not torch.isfinite(expected).all().item():
        raise AssertionError(f"{name}: expected tensor contains NaN or Inf")
    print(f"    {name:16s} {difference_summary(actual, expected)}")
    torch.testing.assert_close(
        actual.detach().float(),
        expected.detach().float(),
        rtol=rtol,
        atol=atol,
        msg=lambda msg: f"{name} mismatch:\n{msg}",
    )


def check_provider_correctness(
    reference: Provider,
    candidate: Provider,
    base_input: torch.Tensor,
    base_grad_output: torch.Tensor,
    rtol: float,
    atol: float,
) -> None:
    reference.module.zero_grad(set_to_none=True)
    candidate.module.zero_grad(set_to_none=True)

    reference_input = base_input.detach().clone().requires_grad_(True)
    candidate_input = base_input.detach().clone().requires_grad_(True)

    reference_output = reference.module(reference_input)
    candidate_output = candidate.module(candidate_input)

    # The two backward calls must not share grad_output: Liger in-place
    # backward is allowed to overwrite its dY buffer with dX.
    reference_output.backward(base_grad_output.detach().clone())
    candidate_output.backward(base_grad_output.detach().clone())
    torch.npu.synchronize()

    print(f"  {candidate.name}")
    assert_tensor_close("forward", candidate_output, reference_output, rtol, atol)
    assert_tensor_close("input gradient", candidate_input.grad, reference_input.grad, rtol, atol)
    assert_tensor_close(
        "weight gradient",
        candidate.module.weight.grad,
        reference.module.weight.grad,
        rtol,
        atol,
    )

    reference.module.zero_grad(set_to_none=True)
    candidate.module.zero_grad(set_to_none=True)


def reset_peak_memory_stats(device: torch.device) -> None:
    try:
        torch.npu.reset_peak_memory_stats(device)
    except TypeError:
        torch.npu.reset_peak_memory_stats()


def max_memory_allocated(device: torch.device) -> int:
    try:
        return int(torch.npu.max_memory_allocated(device))
    except TypeError:
        return int(torch.npu.max_memory_allocated())


def memory_allocated(device: torch.device) -> int:
    try:
        return int(torch.npu.memory_allocated(device))
    except TypeError:
        return int(torch.npu.memory_allocated())


def run_forward(module: nn.Module, input_tensor: torch.Tensor) -> None:
    with torch.no_grad():
        module(input_tensor)


def run_forward_backward(
    module: nn.Module,
    input_tensor: torch.Tensor,
    base_grad_output: torch.Tensor,
) -> None:
    input_tensor.grad = None
    module.zero_grad(set_to_none=True)
    output = module(input_tensor)
    output.backward(base_grad_output.detach().clone())


def benchmark_provider(
    provider: Provider,
    mode: str,
    input_tensor: torch.Tensor,
    base_grad_output: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> BenchmarkResult:
    provider.module.train(mode == "forward-backward")

    if mode == "forward":
        operation: Callable[[], None] = lambda: run_forward(provider.module, input_tensor)
    else:
        operation = lambda: run_forward_backward(provider.module, input_tensor, base_grad_output)

    for _ in range(warmup):
        operation()
    torch.npu.synchronize()

    input_tensor.grad = None
    provider.module.zero_grad(set_to_none=True)
    torch.npu.empty_cache()
    torch.npu.synchronize()

    sample_times_ms: list[float] = []
    peak_memory_bytes = 0
    for _ in range(repeats):
        reset_peak_memory_stats(device)
        starting_memory = memory_allocated(device)

        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start

        sample_times_ms.append(elapsed * 1000.0 / iterations)
        peak_memory_bytes = max(
            peak_memory_bytes,
            max_memory_allocated(device) - starting_memory,
        )

        input_tensor.grad = None
        provider.module.zero_grad(set_to_none=True)

    return BenchmarkResult(
        provider=provider.name,
        mode=mode,
        iteration_ms=statistics.median(sample_times_ms),
        samples_ms=sample_times_ms,
        peak_memory_mib=max(0, peak_memory_bytes) / (1024.0**2),
    )


def print_results(results: list[BenchmarkResult]) -> None:
    print("\nSteady-state benchmark")
    print(
        f"  {'mode':18s} {'provider':22s} {'median ms':>12s} "
        f"{'speedup':>10s} {'peak MiB':>12s} {'repeat samples (ms)'}"
    )

    baseline_by_mode = {
        result.mode: result.iteration_ms
        for result in results
        if result.provider == "pytorch_qwen3"
    }
    for result in results:
        baseline = baseline_by_mode[result.mode]
        speedup = baseline / result.iteration_ms
        samples = ", ".join(f"{sample:.4f}" for sample in result.samples_ms)
        print(
            f"  {result.mode:18s} {result.provider:22s} "
            f"{result.iteration_ms:12.4f} {speedup:9.3f}x "
            f"{result.peak_memory_mib:12.2f} {samples}"
        )


def main() -> None:
    args = parse_args()
    ensure_runtime_support()

    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    torch.npu.set_device(args.device)

    device = torch.device(f"npu:{args.device}")
    dtype = DTYPES[args.dtype]
    shape = (args.batch_size, args.sequence_length, args.hidden_size)

    initial_weight = torch.empty(args.hidden_size, device=device, dtype=dtype)
    initial_weight.uniform_(0.8, 1.2)
    base_input = torch.randn(shape, device=device, dtype=dtype)
    base_grad_output = torch.randn(shape, device=device, dtype=dtype)

    providers = make_providers(
        hidden_size=args.hidden_size,
        eps=args.eps,
        device=device,
        dtype=dtype,
        initial_weight=initial_weight,
    )

    print("Environment")
    print(f"  torch:          {torch.__version__}")
    print(f"  torch_npu:      {package_version('torch-npu')}")
    print(f"  triton-ascend:  {package_version('triton-ascend')}")
    print(f"  liger-kernel:   {package_version('liger-kernel')}")
    print(f"  device:         {device}")
    print(f"  dtype:          {dtype}")
    print(f"  input shape:    {shape}")
    print(f"  warmup:         {args.warmup}")
    print(f"  iterations:     {args.iterations}")
    print(f"  repeats:        {args.repeats}")
    try:
        print(f"  native schema:  {torch.ops.npu.npu_rms_norm.default._schema}")
    except Exception as exc:
        print(f"  native schema:  unavailable ({exc!r})")

    if args.check_correctness:
        print("\nCorrectness check against pytorch_qwen3")
        reference = providers[0]
        print("  pytorch_qwen3 (reference)")
        for candidate in providers[1:]:
            check_provider_correctness(
                reference=reference,
                candidate=candidate,
                base_input=base_input,
                base_grad_output=base_grad_output,
                rtol=args.rtol,
                atol=args.atol,
            )
        print("  PASS: all providers match the PyTorch reference.")

    modes = ("forward", "forward-backward") if args.mode == "all" else (args.mode,)
    results: list[BenchmarkResult] = []
    for mode in modes:
        input_tensor = base_input.detach().clone().requires_grad_(mode == "forward-backward")
        for provider in providers:
            print(f"\nBenchmarking mode={mode}, provider={provider.name} ...")
            result = benchmark_provider(
                provider=provider,
                mode=mode,
                input_tensor=input_tensor,
                base_grad_output=base_grad_output,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            results.append(result)
            print(
                f"  median={result.iteration_ms:.4f} ms, "
                f"incremental_peak={result.peak_memory_mib:.2f} MiB"
            )

    print_results(results)
    print(
        "\nNote: compare steady-state medians, not the first Triton invocation. "
        "End-to-end NanoVLM throughput must still be measured separately."
    )


if __name__ == "__main__":
    main()
