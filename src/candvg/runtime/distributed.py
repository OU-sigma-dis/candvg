"""Runtime helpers for single-process and torch.distributed execution."""

import os
from dataclasses import dataclass
from datetime import timedelta

import torch


@dataclass(frozen=True)
class RuntimeContext:
    """Runtime state shared across distributed and single-GPU modes."""

    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def _get_int_env(name: str, default: int) -> int:
    """Read an integer environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_distributed_timeout() -> timedelta:
    """Read the distributed process-group timeout from environment."""
    seconds = _get_int_env("TORCH_DISTRIBUTED_TIMEOUT_SECONDS", 10000)
    return timedelta(seconds=seconds)


def _set_env_default(name: str, value: int | str) -> None:
    """Set an environment variable only when it is currently unset."""
    os.environ.setdefault(name, str(value))


def _prepare_distributed_env(use_distributed: bool) -> None:
    """Populate torch.distributed env vars from Slurm when needed."""
    if not use_distributed:
        return

    if "SLURM_PROCID" in os.environ:
        _set_env_default("RANK", os.environ["SLURM_PROCID"])
        _set_env_default("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
        _set_env_default("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))

    _set_env_default("RANK", 0)
    _set_env_default("WORLD_SIZE", 1)
    _set_env_default("LOCAL_RANK", 0)
    _set_env_default("MASTER_PORT", 29500)

    if "MASTER_ADDR" not in os.environ:
        slurm_nodes = _get_int_env("SLURM_NNODES", 1)
        if slurm_nodes > 1:
            raise RuntimeError(
                "MASTER_ADDR is required for multi-node Slurm jobs. "
                "Set it in the sbatch script before launching srun."
            )
        os.environ["MASTER_ADDR"] = "127.0.0.1"


def resolve_runtime(mode: str) -> RuntimeContext:
    """Resolve runtime mode and initialize distributed process group if needed."""
    world_size = _get_int_env("WORLD_SIZE", _get_int_env("SLURM_NTASKS", 1))
    rank = _get_int_env("RANK", _get_int_env("SLURM_PROCID", 0))
    local_rank = _get_int_env("LOCAL_RANK", _get_int_env("SLURM_LOCALID", 0))

    if mode == "single":
        use_distributed = False
    elif mode == "distributed":
        use_distributed = True
    else:
        use_distributed = world_size > 1

    _prepare_distributed_env(use_distributed)
    world_size = _get_int_env("WORLD_SIZE", world_size)
    rank = _get_int_env("RANK", rank)
    local_rank = _get_int_env("LOCAL_RANK", local_rank)

    if use_distributed and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
            timeout=_get_distributed_timeout(),
        )

    if use_distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_rank = int(os.getenv("LOCAL_RANK", str(rank)))

    if torch.cuda.is_available():
        if use_distributed:
            device_index = local_rank % torch.cuda.device_count()
            torch.cuda.set_device(device_index)
            device = torch.device("cuda", device_index)
        else:
            device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    return RuntimeContext(
        distributed=use_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def cleanup_runtime(runtime: RuntimeContext, barrier: bool = True) -> None:
    """Synchronize and tear down distributed process groups if enabled."""
    if runtime.distributed and torch.distributed.is_initialized():
        if barrier:
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()
