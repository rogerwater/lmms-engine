import json
import os
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Optional

import torch
from loguru import logger
from torch import profiler as torch_profiler


class StepProfiler:
    """Collect a bounded CPU/CUDA/NPU operator trace.

    ``start_step`` and ``end_step`` count calls to :meth:`step`, which currently
    correspond to training micro-steps. NPU traces are exported and analysed by
    ``torch_npu.profiler.tensorboard_trace_handler``; PyTorch CPU/CUDA traces
    retain the existing Chrome trace output.
    """

    def __init__(
        self,
        enable: bool,
        directory: str,
        rank: int = 0,
        profiler_config: Optional[Dict[str, Any]] = None,
    ):
        self.enable = enable
        self.rank = rank
        self.directory = os.path.abspath(directory)
        self.profiler_config = profiler_config or {}
        self.prof = None
        self.skip_prof = True
        self.backend = "disabled"
        self.steps_seen = 0

        if not self.enable:
            return

        self.start_step = int(self.profiler_config.get("start_step", 0))
        self.end_step = int(self.profiler_config.get("end_step", 5))
        self.warmup_steps = int(self.profiler_config.get("warmup_steps", 1))
        if self.start_step < 0:
            raise ValueError("profiler_config.start_step must be non-negative")
        if self.end_step <= self.start_step:
            raise ValueError("profiler_config.end_step must be greater than start_step")
        if self.warmup_steps < 0:
            raise ValueError("profiler_config.warmup_steps must be non-negative")

        profiler_module = torch_profiler
        activities = [torch_profiler.ProfilerActivity.CPU]
        npu_is_available = False
        torch_npu_module = getattr(torch, "npu", None)
        if torch_npu_module is not None:
            try:
                npu_is_available = bool(torch_npu_module.is_available())
            except (AttributeError, RuntimeError):
                npu_is_available = False

        if npu_is_available:
            try:
                import torch_npu as torch_npu_import

                profiler_module = torch_npu_import.profiler
                activities = [
                    profiler_module.ProfilerActivity.CPU,
                    profiler_module.ProfilerActivity.NPU,
                ]
                self.backend = "npu"
            except ImportError:
                logger.exception("[Profiler] NPU is available but torch_npu could not be imported; using CPU profiling")
                self.backend = "cpu"
        elif torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
            self.backend = "cuda"
        else:
            self.backend = "cpu"

        configured_ranks = self.profiler_config.get("ranks")
        if configured_ranks is None and self.backend == "npu":
            configured_ranks = [0]
        if isinstance(configured_ranks, str) and configured_ranks.lower() == "all":
            self.ranks = None
        elif configured_ranks is None:
            self.ranks = None
        elif isinstance(configured_ranks, int):
            self.ranks = {configured_ranks}
        else:
            self.ranks = {int(configured_rank) for configured_rank in configured_ranks}
        if self.ranks is not None and self.rank not in self.ranks:
            logger.info(f"[Profiler] disabled on rank {self.rank}; configured ranks are {sorted(self.ranks)}")
            return

        os.makedirs(self.directory, exist_ok=True)
        self.activities = activities
        active_steps = self.end_step - self.start_step
        schedule = profiler_module.schedule(
            wait=self.start_step,
            warmup=self.warmup_steps,
            active=active_steps,
            repeat=1,
        )
        self.total_scheduled_steps = self.start_step + self.warmup_steps + active_steps

        record_shapes = bool(self.profiler_config.get("record_shapes", True))
        profile_memory = bool(self.profiler_config.get("profile_memory", self.backend != "npu"))
        with_stack = bool(self.profiler_config.get("with_stack", self.backend != "npu"))
        profile_kwargs = {
            "activities": activities,
            "schedule": schedule,
            "record_shapes": record_shapes,
            "profile_memory": profile_memory,
            "with_stack": with_stack,
            "with_flops": bool(self.profiler_config.get("with_flops", False)),
            "with_modules": bool(self.profiler_config.get("with_modules", False)),
        }
        if self.backend == "npu":
            profile_kwargs["on_trace_ready"] = profiler_module.tensorboard_trace_handler(
                self.directory,
                worker_name=f"rank_{self.rank}",
                analyse_flag=bool(self.profiler_config.get("analyse_flag", True)),
                async_mode=bool(self.profiler_config.get("async_mode", False)),
            )

        self.prof = profiler_module.profile(**profile_kwargs)
        self.skip_prof = False
        logger.info(
            f"[Profiler] configured backend={self.backend}, rank={self.rank}, "
            f"start_step={self.start_step}, warmup_steps={self.warmup_steps}, "
            f"end_step={self.end_step}, output={self.directory}"
        )

    def check(self):
        return self.prof is not None and not self.skip_prof

    def start(self):
        if self.check():
            logger.info(f"[Profiler] started for rank {self.rank}")
            self.prof.start()

    def step(self):
        if self.check():
            self.prof.step()
            self.steps_seen += 1

    def stop(self):
        if self.check():
            logger.info(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()

    def save(self):
        if self.prof is not None:
            if self.backend == "npu":
                logger.info(
                    f"[Profiler] NPU trace for rank {self.rank} was exported under {self.directory} "
                    "by tensorboard_trace_handler"
                )
            else:
                os.makedirs(self.directory, exist_ok=True)
                save_file_name = f"prof_start_{self.start_step}_end_{self.end_step}_rank_{self.rank}.json"
                save_path = os.path.join(self.directory, save_file_name)
                logger.info(f"[Profiler] Saving trace to {save_path}")
                self.prof.export_chrome_trace(save_path)
            self.skip_prof = True

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()

    def should_save(self, step: Optional[int] = None):
        if self.check():
            return self.steps_seen >= self.total_scheduled_steps
        else:
            return False

    def stop_trace(self):
        if self.check():
            logger.info(f"[Profiler] Trace stopped for rank {self.rank}")
            self.skip_prof = True


class MemorySnapshotProfiler:
    """CUDA memory snapshot profiler with automatic OOM capture.

    When enabled, records every CUDA alloc/free event (with Python stack
    traces) into an in-memory ring buffer via
    ``torch.cuda.memory._record_memory_history``. On a ``CUDA OOM``, an
    out-of-memory observer dumps the buffer to a ``.pickle`` file that can
    be loaded into https://pytorch.org/memory_viz for visualization.

    Independent of ``StepProfiler`` — both can be enabled together.

    Config keys (under ``memory_snapshot_config``):
        - ``max_entries`` (int, default 100000): ring buffer size. One alloc
          or free event = one entry. 100k covers ~a few training steps.
        - ``stop_step`` (int, optional): if set, stop recording and dump a
          final snapshot at this global step (useful for inspecting steady
          state without OOM).
    """

    def __init__(
        self,
        enable: bool,
        directory: str,
        rank: int = 0,
        memory_snapshot_config: Optional[Dict[str, Any]] = None,
    ):
        self.enable = enable and torch.cuda.is_available()
        self.rank = rank
        self.directory = directory
        self.config = memory_snapshot_config or {}
        self.max_entries = int(self.config.get("max_entries", 100000))
        self.stop_step = self.config.get("stop_step", None)
        self.started = False
        self.stopped = False

    def _dump(self, filename: str, force: bool = False):
        if not self.enable or (self.stopped and not force):
            return
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, filename)
        try:
            torch.cuda.memory._dump_snapshot(path)
            logger.info(f"[MemSnapshot] dumped snapshot to {path} (rank {self.rank})")
        except Exception:
            logger.exception(f"[MemSnapshot] failed to dump snapshot to {path}")

    def dump_on_exception(self, reason: str):
        timestamp = int(time.time())
        self._dump(f"snapshot_{reason}_rank{self.rank}_pid{os.getpid()}_{timestamp}.pickle", force=True)

    def _oom_observer(self, device, alloc, device_alloc, device_free):
        # Called by PyTorch BEFORE raising CUDA OOM. Dump current snapshot.
        logger.error(
            f"[MemSnapshot] CUDA OOM on rank {self.rank} device {device}: "
            f"attempted to alloc {alloc} bytes "
            f"(device_alloc={device_alloc}, device_free={device_free})"
        )
        self.dump_on_exception("oom_observer")
        # Mark stopped so we don't try to dump again on re-raise paths.
        self.stopped = True

    def start(self):
        if not self.enable or self.started:
            return
        os.makedirs(self.directory, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
        try:
            torch._C._cuda_attach_out_of_memory_observer(self._oom_observer)
        except AttributeError:
            logger.warning(
                "[MemSnapshot] OOM observer API not available in this torch version; "
                "snapshot will only dump on explicit stop_and_save()."
            )
        self.started = True
        logger.info(
            f"[MemSnapshot] recording started on rank {self.rank} "
            f"(max_entries={self.max_entries}, dir={self.directory})"
        )

    def step(self, global_step: int):
        """Mark step boundary in the snapshot timeline; optionally auto-stop."""
        if not self.enable or self.stopped:
            return
        # NVTX marker → shows up as a vertical line in memory_viz timeline.
        torch.cuda.nvtx.range_push(f"step_{global_step}")
        torch.cuda.nvtx.range_pop()
        if self.stop_step is not None and global_step >= self.stop_step:
            self.stop_and_save(reason="stop_step")

    def stop_and_save(self, reason: str = "manual"):
        if not self.enable or self.stopped:
            return
        self._dump(f"snapshot_{reason}_rank{self.rank}.pickle")
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception:
            pass
        self.stopped = True
        logger.info(f"[MemSnapshot] recording stopped (reason={reason}, rank {self.rank})")


class CudaEventProfiler:
    """Low-overhead CUDA event profiler for long-running distributed jobs.

    Unlike torch.profiler, this profiler only records named CUDA event pairs and
    writes completed durations to JSONL. Non-blocking flushes avoid introducing
    synchronization into the training step.
    """

    def __init__(
        self,
        enable: bool,
        directory: str,
        rank: int = 0,
        profiler_config: Optional[Dict[str, Any]] = None,
    ):
        self.enable = enable and torch.cuda.is_available()
        self.rank = rank
        self.directory = directory
        self.profiler_config = profiler_config or {}
        self.start_step = self.profiler_config.get("start_step", 0)
        self.end_step = self.profiler_config.get("end_step")
        self.record_every_n_steps = max(int(self.profiler_config.get("record_every_n_steps", 10)), 1)
        self.flush_every_n_steps = max(int(self.profiler_config.get("flush_every_n_steps", 10)), 1)
        self.ranks = self.profiler_config.get("ranks")
        if self.ranks is not None:
            self.ranks = {int(rank) for rank in self.ranks}
        self.pending_events = []
        self._last_flush_step = -1
        self._file = None

        if not self.enable:
            if enable:
                logger.warning("[CudaEventProfiler] CUDA is unavailable; profiler is disabled")
            return
        if self.ranks is not None and self.rank not in self.ranks:
            self.enable = False
            return

        os.makedirs(self.directory, exist_ok=True)
        self.path = os.path.join(self.directory, f"cuda_events_rank_{self.rank}.jsonl")
        self._file = open(self.path, "a", buffering=1)
        logger.info(f"[CudaEventProfiler] Writing CUDA event timings to {self.path}")

    def should_record(self, step: int) -> bool:
        if not self.enable:
            return False
        if self.ranks is not None and self.rank not in self.ranks:
            return False
        if step < self.start_step:
            return False
        if self.end_step is not None and step > self.end_step:
            return False
        return (step - self.start_step) % self.record_every_n_steps == 0

    @contextmanager
    def record(self, name: str, step: int, **metadata):
        if not self.should_record(step):
            yield
            return

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        try:
            yield
        finally:
            end_event.record()
            self.pending_events.append(
                {
                    "name": name,
                    "step": step,
                    "rank": self.rank,
                    "start_event": start_event,
                    "end_event": end_event,
                    "metadata": metadata,
                }
            )

    def maybe_flush(self, step: int):
        if not self.enable:
            return
        if step == self._last_flush_step:
            return
        if step % self.flush_every_n_steps != 0:
            return
        self.flush(blocking=False)
        self._last_flush_step = step

    def flush(self, blocking: bool = False):
        if not self.enable or self._file is None:
            return

        remaining_events = []
        for event in self.pending_events:
            end_event = event["end_event"]
            if blocking:
                end_event.synchronize()
            elif not end_event.query():
                remaining_events.append(event)
                continue

            record = {
                "name": event["name"],
                "step": event["step"],
                "rank": event["rank"],
                "duration_ms": event["start_event"].elapsed_time(end_event),
            }
            if event["metadata"]:
                record.update(event["metadata"])
            self._file.write(json.dumps(record, sort_keys=True) + "\n")

        self.pending_events = remaining_events

    def close(self):
        if not self.enable:
            return
        self.flush(blocking=True)
        if self._file is not None:
            self._file.close()
            self._file = None
