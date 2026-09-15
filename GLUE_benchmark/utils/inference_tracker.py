#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU Inference Tracker — tracks per-sample inference metrics:
time, latency, VRAM usage, power draw, and energy consumption.

Used by both the post-training inference pass in BaseTrainer and the
standalone evaluate_glue.py script.
"""

import time
import threading
import subprocess
from typing import List, Dict, Optional, Any

import torch

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False


class GPUInferenceTracker:
    """Tracks per-sample and aggregate inference metrics: time, power, energy, VRAM.

    Separates **total time** (tokenization + forward pass + postprocessing)
    from **forward time** (model forward/predict only).

    Uses pynvml when available; falls back to nvidia-smi subprocess calls.
    Spawns a lightweight background thread that polls GPU power/memory at a
    configurable interval while a sample is being processed.
    """

    def __init__(self, gpu_index: int = 0, poll_interval: float = 0.05):
        self.gpu_index = gpu_index
        self.poll_interval = poll_interval
        self._handle = None
        if PYNVML_AVAILABLE:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                pass

        # Per-sample scratch
        self._sample_power_readings: List[float] = []
        self._sample_vram_readings: List[float] = []
        self._sample_start: float = 0.0
        self._fwd_start: float = 0.0
        self._fwd_elapsed: float = 0.0
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None

        # Aggregate accumulators
        self.sample_total_times: List[float] = []
        self.sample_fwd_times: List[float] = []
        self.sample_num_items: List[int] = []
        self.sample_energy: List[float] = []
        self.sample_avg_power: List[float] = []
        self.sample_peak_vram: List[float] = []

    # ---- low-level GPU readers -------------------------------------------

    def _read_power_watts(self) -> Optional[float]:
        """Return current GPU power draw in watts."""
        if self._handle is not None:
            try:
                return pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            except pynvml.NVMLError:
                return None
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 f"--id={self.gpu_index}",
                 "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                timeout=2,
            )
            return float(out.decode().strip())
        except Exception:
            return None

    def _read_vram_mb(self) -> Optional[float]:
        """Return current GPU memory used in MiB."""
        if self._handle is not None:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return info.used / (1024 ** 2)
            except pynvml.NVMLError:
                return None
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 f"--id={self.gpu_index}",
                 "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                timeout=2,
            )
            return float(out.decode().strip())
        except Exception:
            return None

    # ---- background poller -----------------------------------------------

    def _poll_loop(self):
        while self._polling:
            pw = self._read_power_watts()
            vr = self._read_vram_mb()
            if pw is not None:
                self._sample_power_readings.append(pw)
            if vr is not None:
                self._sample_vram_readings.append(vr)
            time.sleep(self.poll_interval)

    # ---- public API -------------------------------------------------------

    def start_sample(self):
        """Call at the beginning of processing a sample."""
        self._sample_power_readings = []
        self._sample_vram_readings = []
        self._fwd_elapsed = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._sample_start = time.perf_counter()
        self._polling = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def start_forward(self):
        """Call right before model forward pass."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._fwd_start = time.perf_counter()

    def end_forward(self, num_items: int = 1):
        """Call right after model forward pass."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._fwd_elapsed = time.perf_counter() - self._fwd_start
        self.sample_fwd_times.append(self._fwd_elapsed)
        self.sample_num_items.append(num_items)

    def end_sample(self):
        """Call at the end of processing a sample."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_elapsed = time.perf_counter() - self._sample_start
        self._polling = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)

        self.sample_total_times.append(total_elapsed)

        if self._sample_power_readings:
            avg_pw = sum(self._sample_power_readings) / len(self._sample_power_readings)
            energy_j = avg_pw * total_elapsed
        else:
            avg_pw = 0.0
            energy_j = 0.0
        self.sample_avg_power.append(avg_pw)
        self.sample_energy.append(energy_j)

        peak_vram = max(self._sample_vram_readings) if self._sample_vram_readings else 0.0
        self.sample_peak_vram.append(peak_vram)

    def summary(self) -> Dict[str, Any]:
        """Return aggregate inference statistics."""
        n = len(self.sample_total_times)
        if n == 0:
            return {}

        total_time = sum(self.sample_total_times)
        total_fwd_time = sum(self.sample_fwd_times) if self.sample_fwd_times else 0.0
        total_items = sum(self.sample_num_items) if self.sample_num_items else 0

        throughput = (total_items / total_fwd_time) if total_fwd_time > 0 else 0.0
        avg_latency = (total_fwd_time / n) if n > 0 else 0.0

        return {
            "num_samples": n,
            # End-to-end timing
            "total_time_s": round(total_time, 3),
            "avg_time_per_sample_s": round(total_time / n, 4),
            # Forward-only timing
            "total_forward_time_s": round(total_fwd_time, 3),
            "avg_forward_time_per_sample_s": round(total_fwd_time / n, 4) if n > 0 else 0.0,
            # Inference latency = avg forward time per sample
            "inference_latency_s": round(avg_latency, 4),
            # Throughput
            "total_items_processed": total_items,
            "throughput_samples_per_s": round(throughput, 2),
            # Energy
            "total_energy_j": round(sum(self.sample_energy), 3),
            "avg_energy_per_sample_j": round(sum(self.sample_energy) / n, 4),
            "total_energy_wh": round(sum(self.sample_energy) / 3600.0, 6),
            # Power
            "avg_power_w": round(sum(self.sample_avg_power) / n, 2),
            # VRAM
            "peak_vram_mb": round(max(self.sample_peak_vram), 2) if self.sample_peak_vram else 0.0,
            "avg_peak_vram_per_sample_mb": round(sum(self.sample_peak_vram) / n, 2),
            # Per-sample detail
            "per_sample_total_time_s": [round(t, 4) for t in self.sample_total_times],
            "per_sample_forward_time_s": [round(t, 4) for t in self.sample_fwd_times],
            "per_sample_energy_j": [round(e, 4) for e in self.sample_energy],
            "per_sample_avg_power_w": [round(p, 2) for p in self.sample_avg_power],
            "per_sample_peak_vram_mb": [round(v, 2) for v in self.sample_peak_vram],
        }
