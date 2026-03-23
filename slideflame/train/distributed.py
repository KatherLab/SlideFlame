"""
distributed.py

Clean distributed utilities for:
  - single GPU / single process (no torch.distributed init)
  - torchrun / multi-GPU DDP (init_process_group)

Fixes vs your current version:
  1) Does NOT call init_process_group in single-GPU mode (your current code does).
     That behavior is a common source of hangs / env:// errors when MASTER_ADDR/PORT
     aren't set.
  2) Sets CUDA device correctly in distributed mode.
  3) Chooses a sane backend fallback if CUDA isn't available.
"""

import os
import torch


def is_master(args, local: bool = False) -> bool:
    if local:
        return getattr(args, "local_rank", 0) == 0
    return getattr(args, "rank", 0) == 0


def is_using_distributed() -> bool:
    # torchrun sets WORLD_SIZE, RANK, LOCAL_RANK
    ws = os.environ.get("WORLD_SIZE", None)
    if ws is not None:
        try:
            return int(ws) > 1
        except ValueError:
            return False

    # SLURM fallback
    ntasks = os.environ.get("SLURM_NTASKS", None)
    if ntasks is not None:
        try:
            return int(ntasks) > 1
        except ValueError:
            return False

    return False


def world_info_from_env():
    """
    Returns (local_rank, global_rank, world_size)
    Works for torchrun, MPI, SLURM.
    """
    local_rank = 0
    for v in ("LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break

    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break

    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    """
    Initializes torch.distributed only if truly running multi-process.

    Side effects:
      - populates args.distributed, args.rank, args.local_rank, args.world_size, args.device
      - sets CUDA device if available
      - returns torch.device
    """
    using_dist = is_using_distributed()
    args.distributed = using_dist

    if using_dist:
        args.local_rank, args.rank, args.world_size = world_info_from_env()

        # Set device before init_process_group (good practice)
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            device = torch.device(f"cuda:{args.local_rank}")
            backend = getattr(args, "dist_backend", "nccl")
        else:
            device = torch.device("cpu")
            backend = "gloo"

        dist_url = getattr(args, "dist_url", "env://")
        torch.distributed.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    else:
        # Single process / single GPU: DO NOT init process group.
        args.local_rank, args.rank, args.world_size = 0, 0, 1

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    args.device = str(device)
    return device