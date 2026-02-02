#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NVML-based callback for tracking GPU power and VRAM usage during training.

IMPORTANT: This implementation samples during actual training computation,
not just at checkpoint saves (which would capture I/O power, not training power).
"""

import os
import json
import time
import threading
import statistics
from typing import Optional, List, Dict, Any

import torch
from transformers import TrainerCallback

# ---- NVML (NVIDIA Management Library) import ----
# Prefer nvidia-ml-py (newer) over pynvml (deprecated)
NVML_OK = False
nvmlInit = None
nvmlShutdown = None
nvmlDeviceGetHandleByIndex = None
nvmlDeviceGetPowerUsage = None
nvmlDeviceGetMemoryInfo = None

try:
    from pynvml import (
        nvmlInit as _nvmlInit,
        nvmlShutdown as _nvmlShutdown,
        nvmlDeviceGetHandleByIndex as _nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage as _nvmlDeviceGetPowerUsage,
        nvmlDeviceGetMemoryInfo as _nvmlDeviceGetMemoryInfo,
    )
    nvmlInit = _nvmlInit
    nvmlShutdown = _nvmlShutdown
    nvmlDeviceGetHandleByIndex = _nvmlDeviceGetHandleByIndex
    nvmlDeviceGetPowerUsage = _nvmlDeviceGetPowerUsage
    nvmlDeviceGetMemoryInfo = _nvmlDeviceGetMemoryInfo
    NVML_OK = True
except Exception:
    try:
        from nvidia_ml_py import (
            nvmlInit as _nvmlInit,
            nvmlShutdown as _nvmlShutdown,
            nvmlDeviceGetHandleByIndex as _nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetPowerUsage as _nvmlDeviceGetPowerUsage,
            nvmlDeviceGetMemoryInfo as _nvmlDeviceGetMemoryInfo,
        )
        nvmlInit = _nvmlInit
        nvmlShutdown = _nvmlShutdown
        nvmlDeviceGetHandleByIndex = _nvmlDeviceGetHandleByIndex
        nvmlDeviceGetPowerUsage = _nvmlDeviceGetPowerUsage
        nvmlDeviceGetMemoryInfo = _nvmlDeviceGetMemoryInfo
        NVML_OK = True
    except Exception:
        NVML_OK = False


class ContinuousNVMLSampler:
    """
    Background thread that continuously samples GPU power at regular intervals.
    This captures actual training power, not just I/O power during checkpoints.
    """
    
    def __init__(self, gpu_index: int = 0, sample_interval_ms: int = 100):
        """
        Args:
            gpu_index: GPU device index
            sample_interval_ms: Sampling interval in milliseconds (default 100ms = 10 samples/sec)
        """
        self.gpu_index = gpu_index
        self.sample_interval = sample_interval_ms / 1000.0  # Convert to seconds
        
        self.samples: List[Dict[str, Any]] = []
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._handle = None
        self.nvml_ok = False
        
    def _nvml_read(self) -> Optional[Dict[str, float]]:
        """Read power and memory from NVML."""
        try:
            if self._handle is None:
                return None
            power_w = nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            mem = nvmlDeviceGetMemoryInfo(self._handle)
            return {
                "power_w": power_w,
                "vram_used_mb": float(mem.used) / (1024 ** 2),
                "vram_total_mb": float(mem.total) / (1024 ** 2),
            }
        except Exception:
            return None
    
    def _sampling_loop(self):
        """Background sampling loop."""
        while self.is_running:
            sample = self._nvml_read()
            if sample:
                sample["timestamp"] = time.time()
                with self._lock:
                    self.samples.append(sample)
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start continuous sampling."""
        if not NVML_OK:
            return False
        
        try:
            nvmlInit()
            self._handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
            self.nvml_ok = True
        except Exception:
            self.nvml_ok = False
            return False
        
        self.is_running = True
        self.samples = []
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()
        return True
    
    def stop(self) -> List[Dict[str, Any]]:
        """Stop sampling and return all samples."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        with self._lock:
            samples = self.samples.copy()
        
        try:
            nvmlShutdown()
        except Exception:
            pass
        
        return samples
    
    def get_current_samples(self) -> List[Dict[str, Any]]:
        """Get a copy of current samples without stopping."""
        with self._lock:
            return self.samples.copy()


class CheckpointNVMLCallback(TrainerCallback):
    """
    NVML callback that samples power during actual training, not just at checkpoints.
    
    Features:
    - Continuous background sampling during training (captures compute power)
    - Per-step sampling option (on_step_end)
    - Proper energy calculation via trapezoidal integration
    - Tracks VRAM usage
    - Records PyTorch allocator peaks
    """
    
    def __init__(
        self, 
        track_torch_peaks: bool = True, 
        gpu_index: int = 0,
        sample_interval_ms: int = 100,
        use_background_sampling: bool = True,
        sample_every_n_steps: int = 10,
    ):
        """
        Args:
            track_torch_peaks: Track PyTorch memory allocator peaks
            gpu_index: GPU device index for NVML
            sample_interval_ms: Background sampling interval in ms (default 100ms)
            use_background_sampling: Use continuous background thread sampling
            sample_every_n_steps: Also sample at every N training steps
        """
        self.track_torch_peaks = track_torch_peaks
        self.gpu_index = gpu_index
        self.sample_interval_ms = sample_interval_ms
        self.use_background_sampling = use_background_sampling
        self.sample_every_n_steps = sample_every_n_steps
        
        self.nvml_ok = False
        self._handle = None
        
        # Continuous sampler
        self.continuous_sampler: Optional[ContinuousNVMLSampler] = None
        
        # Step-based samples (during actual training)
        self.step_samples: List[Dict[str, Any]] = []
        
        # Checkpoint samples (kept for compatibility)
        self.checkpoint_samples: List[Dict[str, Any]] = []
        
        # Peak memory tracking
        self.peak_allocated_mb_between_ckpts: List[float] = []
        
        # Timing
        self.train_start_time: Optional[float] = None
        self.train_end_time: Optional[float] = None
        
        # Summary
        self.summary: Dict[str, Any] = {}

    def _nvml_read(self) -> Optional[Dict[str, float]]:
        """Read power and memory from NVML."""
        try:
            if self._handle is None:
                return None
            power_w = nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            mem = nvmlDeviceGetMemoryInfo(self._handle)
            return {
                "power_w": power_w,
                "vram_used_mb": float(mem.used) / (1024 ** 2),
                "vram_total_mb": float(mem.total) / (1024 ** 2),
                "timestamp": time.time(),
            }
        except Exception:
            return None

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize NVML and start sampling on training start."""
        self.train_start_time = time.time()
        self.nvml_ok = False
        
        # Initialize NVML for step-based sampling
        if NVML_OK:
            try:
                nvmlInit()
                self._handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
                self.nvml_ok = True
            except Exception:
                self.nvml_ok = False
        
        # Start continuous background sampling
        if self.use_background_sampling and NVML_OK:
            self.continuous_sampler = ContinuousNVMLSampler(
                gpu_index=self.gpu_index,
                sample_interval_ms=self.sample_interval_ms,
            )
            self.continuous_sampler.start()
        
        # Reset PyTorch memory stats
        if self.track_torch_peaks and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def on_step_end(self, args, state, control, **kwargs):
        """Sample at every N training steps - captures actual training power."""
        if not self.nvml_ok:
            return
        
        # Sample every N steps to reduce overhead
        if state.global_step % self.sample_every_n_steps == 0:
            sample = self._nvml_read()
            if sample:
                sample["global_step"] = state.global_step
                sample["source"] = "training_step"
                self.step_samples.append(sample)

    def on_save(self, args, state, control, **kwargs):
        """Sample at checkpoint save (for compatibility, but marked as I/O)."""
        if self.nvml_ok:
            sample = self._nvml_read()
            if sample:
                sample["global_step"] = state.global_step
                sample["source"] = "checkpoint_save"  # Mark as I/O, not training
                self.checkpoint_samples.append(sample)
        
        # Track PyTorch allocator peak
        if self.track_torch_peaks and torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                peak_bytes = torch.cuda.max_memory_allocated(dev)
                self.peak_allocated_mb_between_ckpts.append(peak_bytes / (1024 ** 2))
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        
        # Persist incrementally
        self._save_timeseries(args.output_dir)

    def on_train_end(self, args, state, control, **kwargs):
        """Compute summary statistics on training end."""
        self.train_end_time = time.time()
        
        # Stop continuous sampler and get all samples
        continuous_samples = []
        if self.continuous_sampler:
            continuous_samples = self.continuous_sampler.stop()
        
        # Compute summary from all sources
        self.summary = self._compute_summary(continuous_samples)
        
        # Save final timeseries with summary
        self._save_timeseries(args.output_dir, final=True)
        
        # Cleanup NVML
        try:
            nvmlShutdown()
        except Exception:
            pass

    def _compute_summary(self, continuous_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics from all sampling sources."""
        summary = {}
        
        # Training duration
        if self.train_start_time and self.train_end_time:
            summary["training_duration_seconds"] = self.train_end_time - self.train_start_time
        
        # === Continuous sampling statistics (most accurate for training power) ===
        if continuous_samples:
            powers = [s["power_w"] for s in continuous_samples]
            vrams = [s["vram_used_mb"] for s in continuous_samples]
            
            summary["continuous_sampling"] = {
                "num_samples": len(continuous_samples),
                "sample_interval_ms": self.sample_interval_ms,
                "avg_power_watts": float(statistics.mean(powers)),
                "max_power_watts": float(max(powers)),
                "min_power_watts": float(min(powers)),
                "std_power_watts": float(statistics.stdev(powers)) if len(powers) > 1 else 0.0,
                "avg_vram_used_mb": float(statistics.mean(vrams)),
                "max_vram_used_mb": float(max(vrams)),
            }
            
            # Calculate energy via trapezoidal integration
            energy_wh = self._calculate_energy_trapezoidal(continuous_samples)
            summary["continuous_sampling"]["energy_wh"] = energy_wh
        
        # === Step-based sampling statistics ===
        if self.step_samples:
            powers = [s["power_w"] for s in self.step_samples]
            vrams = [s["vram_used_mb"] for s in self.step_samples]
            
            summary["step_sampling"] = {
                "num_samples": len(self.step_samples),
                "sample_every_n_steps": self.sample_every_n_steps,
                "avg_power_watts": float(statistics.mean(powers)),
                "max_power_watts": float(max(powers)),
                "min_power_watts": float(min(powers)),
                "avg_vram_used_mb": float(statistics.mean(vrams)),
                "max_vram_used_mb": float(max(vrams)),
            }
            
            # Energy estimate from step sampling
            energy_wh = self._calculate_energy_trapezoidal(self.step_samples)
            summary["step_sampling"]["energy_wh"] = energy_wh
        
        # === Checkpoint sampling statistics (I/O power - less accurate) ===
        if self.checkpoint_samples:
            powers = [s["power_w"] for s in self.checkpoint_samples]
            vrams = [s["vram_used_mb"] for s in self.checkpoint_samples]
            
            summary["checkpoint_sampling"] = {
                "num_samples": len(self.checkpoint_samples),
                "avg_power_watts": float(statistics.mean(powers)),
                "avg_vram_used_mb": float(statistics.mean(vrams)),
                "note": "Captured during I/O (checkpoint saves), not during training compute"
            }
        
        # === Peak memory tracking ===
        if self.peak_allocated_mb_between_ckpts:
            summary["peak_allocator_mb"] = {
                "avg": float(statistics.mean(self.peak_allocated_mb_between_ckpts)),
                "max": float(max(self.peak_allocated_mb_between_ckpts)),
            }
        
        # === Best estimate for energy (prefer continuous, then step, then checkpoint) ===
        if continuous_samples:
            summary["estimated_energy_wh"] = summary["continuous_sampling"]["energy_wh"]
            summary["avg_power_watts"] = summary["continuous_sampling"]["avg_power_watts"]
            summary["energy_source"] = "continuous_sampling"
        elif self.step_samples:
            summary["estimated_energy_wh"] = summary["step_sampling"]["energy_wh"]
            summary["avg_power_watts"] = summary["step_sampling"]["avg_power_watts"]
            summary["energy_source"] = "step_sampling"
        elif self.checkpoint_samples and "training_duration_seconds" in summary:
            # Fallback: estimate from checkpoint samples (less accurate)
            avg_power = summary["checkpoint_sampling"]["avg_power_watts"]
            duration_hours = summary["training_duration_seconds"] / 3600.0
            summary["estimated_energy_wh"] = avg_power * duration_hours
            summary["avg_power_watts"] = avg_power
            summary["energy_source"] = "checkpoint_sampling (fallback, less accurate)"
        
        return summary

    def _calculate_energy_trapezoidal(self, samples: List[Dict[str, Any]]) -> float:
        """
        Calculate energy consumption using trapezoidal integration.
        Energy = ∫Power×dt
        
        Returns energy in Watt-hours (Wh).
        """
        if len(samples) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_samples = sorted(samples, key=lambda x: x["timestamp"])
        
        total_energy_ws = 0.0  # Watt-seconds (Joules)
        
        for i in range(1, len(sorted_samples)):
            dt = sorted_samples[i]["timestamp"] - sorted_samples[i-1]["timestamp"]
            # Trapezoidal rule: average of two power readings × time
            avg_power = (sorted_samples[i]["power_w"] + sorted_samples[i-1]["power_w"]) / 2
            total_energy_ws += avg_power * dt
        
        # Convert Watt-seconds to Watt-hours
        return total_energy_ws / 3600.0

    def _save_timeseries(self, output_dir: str, final: bool = False):
        """Save timeseries data to file."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "power_vram_timeseries.json")
            
            data = {
                "step_samples": self.step_samples,
                "checkpoint_samples": self.checkpoint_samples,
            }
            
            if final:
                data["summary"] = self.summary
                
                # Also include continuous samples in final output
                if self.continuous_sampler:
                    # Don't save all continuous samples (too large), just stats
                    pass
            
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def get_energy_wh(self, training_time_seconds: float = None) -> Optional[float]:
        """Get the best estimate of energy consumption in Wh."""
        return self.summary.get("estimated_energy_wh")
    
    def get_avg_power_watts(self) -> Optional[float]:
        """Get the best estimate of average power in Watts."""
        return self.summary.get("avg_power_watts")


# Backward compatibility alias
NVMLCallback = CheckpointNVMLCallback
