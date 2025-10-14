"""
Common utilities for nanochat.
"""

import os
import re
import logging
from contextlib import nullcontext, AbstractContextManager
from typing import Optional

import torch
import torch.distributed as dist

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def _is_mps_available() -> bool:
    """Return True if PyTorch can use Apple's Metal Performance Shaders backend."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore[attr-defined]

def _select_default_device() -> torch.device:
    """Choose the best available device for single-process execution."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                   █████                 █████
                                                  ░░███                 ░░███
 ████████    ██████   ████████    ██████   ██████  ░███████    ██████   ███████
░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███ ░░░███░
 ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████   ░███
 ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███   ░███ ███
 ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░████████  ░░█████
░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░    ░░░░░
"""
    print0(banner)

def is_ddp():
    # TODO is there a proper way
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def compute_init():
    """Basic initialization that we keep doing over and over, so make common."""

    # Reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    # Treat single-process torchrun (WORLD_SIZE == 1) as non-distributed to avoid unnecessary NCCL setup.
    if ddp and ddp_world_size <= 1:
        ddp = False
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1

    if ddp:
        # DDP currently assumes CUDA/NCCL
        assert torch.cuda.is_available(), "Distributed execution currently requires CUDA."
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device) # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = _select_default_device()
        if device.type == "cuda":
            torch.cuda.set_device(device)

    if ddp_rank == 0:
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            logger.info(f"Distributed world size: {ddp_world_size} | device: CUDA - {props.name}")
        elif device.type == "mps":
            logger.info(f"Distributed world size: {ddp_world_size} | device: MPS (Apple Silicon)")
        else:
            logger.info(f"Distributed world size: {ddp_world_size} | device: CPU")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

def device_synchronize(device: torch.device) -> None:
    """Synchronize the active device if synchronization is supported."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()  # type: ignore[attr-defined]

def reset_peak_memory_stats(device: torch.device) -> None:
    """Reset peak memory stats for the active device when supported."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.reset_peak_memory_stats()  # type: ignore[attr-defined]

def get_peak_memory_bytes(device: torch.device) -> Optional[int]:
    """Return peak memory allocated in bytes if supported by the backend."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device=device)
    if device.type == "mps" and hasattr(torch, "mps"):
        # torch.mps.current_allocated_memory returns current usage; use that as a best-effort proxy.
        return torch.mps.current_allocated_memory()  # type: ignore[attr-defined]
    return None

def autocast_context(*, device: torch.device, dtype: Optional[torch.dtype] = None, enabled: bool = True) -> AbstractContextManager:
    """Return an autocast context suitable for the provided device."""
    if not enabled:
        return nullcontext()

    if device.type == "cuda":
        chosen_dtype = dtype or torch.bfloat16
        return torch.amp.autocast(device_type="cuda", dtype=chosen_dtype)

    if device.type == "cpu":
        chosen_dtype = dtype or torch.bfloat16
        return torch.amp.autocast(device_type="cpu", dtype=chosen_dtype)

    if device.type == "mps":
        chosen_dtype = dtype or torch.float32
        if chosen_dtype != torch.float32:
            logger.warning("MPS backend forces autocast to fp32 for stability; ignoring requested dtype.")
        return nullcontext()

    return nullcontext()

def preferred_autocast_dtype(device: torch.device, requested: Optional[str] = None) -> torch.dtype:
    """Resolve the dtype string used by CLI flags into a torch.dtype compatible with the device."""
    if requested is not None:
        normalized = requested.lower()
        if normalized in {"bf16", "bfloat16"}:
            if device.type == "mps":
                return torch.float16
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half"}:
            if device.type == "cpu":
                return torch.float32
            return torch.float16
        if normalized in {"fp32", "float32"}:
            return torch.float32
    # Default selection by device
    if device.type == "mps":
        return torch.float32
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16
