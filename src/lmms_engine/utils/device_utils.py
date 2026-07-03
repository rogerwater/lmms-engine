import torch

from lmms_engine.utils.import_utils import is_torch_npu_available


def is_npu_available() -> bool:
    return is_torch_npu_available() and hasattr(torch, "npu") and torch.npu.is_available()


def get_accelerator_type() -> str:
    if is_npu_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_current_device() -> torch.device:
    accelerator = get_accelerator_type()
    if accelerator == "npu":
        return torch.device("npu", torch.npu.current_device())
    if accelerator == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def empty_cache() -> None:
    if is_npu_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device_name() -> str:
    if is_npu_available():
        return torch.npu.get_device_name()
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    return "cpu"

