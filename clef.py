"""Load clef-flash locally and answer Jev / SystemOne decision requests."""

import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

MODEL_DIR = Path(os.environ.get("CLEF_MODEL_DIR", Path(__file__).parent / "models" / "clef-flash"))


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name()
    if platform.system() == "Darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            return chip or platform.machine()
        except OSError:
            pass
    return platform.processor() or platform.machine()


class Clef:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str | None = None, optimized: bool = True):
        if not (model_dir / "joint_schema_model.py").exists():
            raise SystemExit(f"model not found in {model_dir}, run: uv run hf download Cloudflare/clef-flash --local-dir {model_dir}")
        sys.path.insert(0, str(model_dir))
        from joint_schema_model import load_release_model, systemone

        self.device = device or pick_device()
        self.hardware = hardware_name(self.device)
        # On CUDA, transformers uses the flash-linear-attention kernels if installed, so leave it alone.
        if optimized and self.device != "cuda":
            from mps_kernels import patch_qwen3_5

            patch_qwen3_5()
        self._systemone = systemone
        self.model, self.processor = load_release_model(model_dir, device=self.device)
        self._lock = threading.Lock()

    def _sync(self) -> None:
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def decide(self, request: dict) -> tuple[dict, float]:
        """Run one SystemOne request. Returns the response and the model latency in milliseconds."""
        with self._lock:
            self._sync()
            start = time.perf_counter()
            response = self._systemone(self.model, self.processor, request)
            self._sync()
            return response, (time.perf_counter() - start) * 1000
