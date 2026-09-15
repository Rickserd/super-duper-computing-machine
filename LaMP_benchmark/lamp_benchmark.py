#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaMP Benchmark Framework - fine-tuning and evaluation in one script.

Fine-tunes small LLMs (Transformers, encoder-decoder, and State-Space Models) on
the LaMP personalization tasks with Contriever profile retrieval, measures GPU
energy / power / VRAM / latency, estimates FLOPs analytically, and scores every
run with NetScore and its efficiency variants (NS, NS-E, NS-M, NS#) for both
the training cost and the inference cost.

Models : flan-t5 (base), tinyllama (1.1B), qwen3 (1.7B), mamba1 (1.4B), mamba2 (1.3B)
Tasks  : lamp1 (citation), lamp2 (movie tags), lamp3 (product rating), lamp4 (headline)
Methods: lora, loraplus, full_ft, bitfit, qlora

Usage (fine-tune + evaluate + NetScore):
    python lamp_benchmark.py train --model_type tinyllama --task lamp2 --method lora \
        --data_dir ./data/dataset2 --output_dir ./outputs/tinyllama_lamp2_lora --bf16

Usage (sweep: comma lists or 'all' for --model_type / --task / --method):
    python lamp_benchmark.py train --model_type qwen3 --task all --method lora,loraplus \
        --data_dir ./data --output_dir ./outputs --bf16

Usage (evaluate a fine-tuned checkpoint):
    python lamp_benchmark.py evaluate --model_type tinyllama --task lamp2 --method lora \
        --data_dir ./data/dataset2 --model_path ./outputs/tinyllama_lamp2_lora/final_model --bf16

Usage (zero-shot evaluation of the base model):
    python lamp_benchmark.py evaluate --zero_shot --model_type qwen3 --task lamp1 \
        --data_dir ./data/dataset1 --output_dir ./zero_shot_results --bf16
"""

import argparse
import gc
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import datasets
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutput
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

# ---- NVML (GPU power / VRAM). Provided by the nvidia-ml-py package. ----
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    pynvml = None
    NVML_OK = False

# ---- mamba_ssm (required only for mamba2) ----
MAMBA_SSM_OK = False
try:
    from dataclasses import fields as dataclass_fields
    from mamba_ssm.models.config_mamba import MambaConfig as _MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
    MAMBA_SSM_OK = True
except ImportError:
    pass

# ---- rouge_score (required only for lamp4) ----
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# =============================================================================
# CONFIGURATION: MODELS, TASKS, METHODS, NETSCORE
# =============================================================================

@dataclass
class ArchSpec:
    """Architecture shapes used by the analytical FLOPs model."""
    family: str  # "transformer" | "t5" | "mamba1" | "mamba2"
    num_layers: int
    hidden_size: int  # d_model
    vocab_size: int = 0  # LM head (included: it is not negligible for generation)
    # Transformer / T5 (num_heads is also the SSM head count for Mamba-2)
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    num_decoder_layers: int = 0  # T5 only
    # Mamba
    d_inner: int = 0
    d_state: int = 0
    d_conv: int = 4
    dt_rank: int = 0
    ngroups: int = 1


@dataclass
class ModelSpec:
    """Configuration for a model type."""
    hf_name: str
    kind: str  # "seq2seq" | "causal"
    lora_targets: List[str]
    lora_task_type: Optional[TaskType]  # None for Mamba (plain PeftModel)
    arch: Optional[ArchSpec] = None
    tokenizer_name: Optional[str] = None
    description: str = ""


MODELS: Dict[str, ModelSpec] = {
    "flan-t5": ModelSpec(
        hf_name="google/flan-t5-base",
        kind="seq2seq",
        lora_targets=["q", "v"],
        lora_task_type=TaskType.SEQ_2_SEQ_LM,
        arch=ArchSpec(
            family="t5", num_layers=12, num_decoder_layers=12, hidden_size=768,
            num_heads=12, head_dim=64, intermediate_size=2048, vocab_size=32128,
        ),
        description="Flan-T5 base (encoder-decoder)",
    ),
    "tinyllama": ModelSpec(
        hf_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        kind="causal",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_task_type=TaskType.CAUSAL_LM,
        arch=ArchSpec(
            family="transformer", num_layers=22, hidden_size=2048, num_heads=32,
            num_kv_heads=4, head_dim=64, intermediate_size=5632, vocab_size=32000,
        ),
        description="TinyLlama 1.1B Chat",
    ),
    "qwen3": ModelSpec(
        hf_name="Qwen/Qwen3-1.7B",
        kind="causal",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_task_type=TaskType.CAUSAL_LM,
        arch=ArchSpec(
            family="transformer", num_layers=28, hidden_size=2048, num_heads=16,
            num_kv_heads=8, head_dim=128, intermediate_size=6144, vocab_size=151936,
        ),
        description="Qwen3 1.7B",
    ),
    "mamba1": ModelSpec(
        hf_name="state-spaces/mamba-1.4b-hf",
        kind="causal",
        lora_targets=["x_proj", "in_proj"],
        lora_task_type=None,
        arch=ArchSpec(
            family="mamba1", num_layers=48, hidden_size=2048, d_inner=4096,
            d_state=16, d_conv=4, dt_rank=128, vocab_size=50280,
        ),
        description="Mamba-1 1.4B (HuggingFace-compatible)",
    ),
    "mamba2": ModelSpec(
        hf_name="state-spaces/mamba2-1.3b",
        kind="causal",
        lora_targets=["in_proj", "x_proj"],  # NOT out_proj/conv1d (fused kernels bypass LoRA)
        lora_task_type=None,
        tokenizer_name="EleutherAI/gpt-neox-20b",
        arch=ArchSpec(
            family="mamba2", num_layers=48, hidden_size=2048, num_heads=64,
            d_inner=4096, d_state=128, d_conv=4, ngroups=1, vocab_size=50288,
        ),
        description="Mamba-2 1.3B (requires mamba_ssm)",
    ),
}
MODEL_ALIASES = {"qwen": "qwen3"}


@dataclass
class TaskConfig:
    """Configuration for a LaMP task."""
    name: str
    dataset_folder: str
    kind: str  # "classification" | "regression" | "generation"
    primary_metric: str
    description: str


TASKS: Dict[str, TaskConfig] = {
    "lamp1": TaskConfig("lamp1", "dataset1", "classification", "accuracy",
                        "Personalized citation identification"),
    "lamp2": TaskConfig("lamp2", "dataset2", "classification", "accuracy",
                        "Personalized movie tagging"),
    "lamp3": TaskConfig("lamp3", "dataset3", "regression", "mae",
                        "Personalized product rating (1-5)"),
    "lamp4": TaskConfig("lamp4", "dataset4", "generation", "rouge1",
                        "Personalized news headline generation"),
}

METHODS = ["lora", "loraplus", "full_ft", "bitfit", "qlora"]
METHOD_NAMES = {"lora": "LoRA", "loraplus": "LoRA+", "full_ft": "Full-FT",
                "bitfit": "BitFit", "qlora": "QLoRA"}
LORA_METHODS = ("lora", "loraplus", "qlora")


@dataclass
class NetScoreConfig:
    """
    NetScore = S * log10( a^alpha / ((p*m)^beta * v^gamma * t^delta * w^lambda) )

    a: task performance, p: trainable parameters, m: FLOPs per sequence,
    v: peak VRAM, t: fine-tuning (or inference) time, w: average power.
    """
    scale: float = 20.0  # S
    alpha: float = 2.0   # Performance exponent
    # Efficiency exponents per variant; non-zero exponents use 1/8 = 0.125
    variants: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "NS":   {"beta": 0.5, "gamma": 0.0,   "delta": 0.0,   "lambda": 0.0},    # params x FLOPs
        "NS-E": {"beta": 0.0, "gamma": 0.0,   "delta": 0.125, "lambda": 0.125},  # energy: time + power
        "NS-M": {"beta": 0.0, "gamma": 0.125, "delta": 0.0,   "lambda": 0.0},    # peak memory
        "NS#":  {"beta": 0.0, "gamma": 0.125, "delta": 0.125, "lambda": 0.125},  # all efficiency terms
    })
    # Units: a = performance * performance_scale; every other raw value is divided by its unit
    performance_scale: float = 100.0  # a in %
    params_unit: float = 1e6          # p in millions
    flops_unit: float = 1e6           # m in millions of FLOPs
    vram_mb_unit: float = 1024.0      # v in GiB (raw values are MiB)
    time_s_unit: float = 1.0          # t in seconds
    power_w_unit: float = 1.0         # w in Watts
    # LaMP-3 is scored by MAE on a 1-5 scale; NetScore performance = 1 - MAE / max_error
    lamp3_max_error: float = 4.0


NETSCORE = NetScoreConfig()


# =============================================================================
# ANALYTICAL FLOPs (the m term of NetScore)
# =============================================================================
#
# For one sequence (S = max_length prompt tokens, T = max_new_tokens):
#
#   F_train = forward FLOPs of the training sequence
#             causal: F(S)            T5: encoder(S) + decoder(T)
#   F_inf   = forward FLOPs of prompt + generated tokens
#             causal: F(S + T)        T5: encoder(S) + num_beams * decoder(T)
#   A       = LoRA adapter forward FLOPs ~ 2 * tokens * trainable_params
#
#   method                 training FLOPs / sequence
#   ---------------------  -------------------------
#   full_ft                3 F_train
#   bitfit                 2 F_train   (bias grads ~0)
#   lora / loraplus        2 F_train + 3A
#   qlora                  2 F_train + 3A  (4-bit saves memory, not matmuls)
#
# Inference is F_inf (adapters assumed merged). A matmul producing an (m x n)
# output from a k-dim contraction costs 2*m*k*n FLOPs. The LM head is included.
# Contriever retrieval is not counted in m (its time/energy is in t and w).
# Models without an ArchSpec fall back to F ~ 2 * total_params * tokens.

def _transformer_forward_flops(a: ArchSpec, S: int) -> float:
    d, hd, d_ff = a.hidden_size, a.head_dim, a.intermediate_size
    q_dim = a.num_heads * hd
    kv_dim = a.num_kv_heads * hd

    attn_proj = 2 * S * d * q_dim          # q_proj
    attn_proj += 2 * 2 * S * d * kv_dim    # k_proj + v_proj (GQA)
    attn_proj += 2 * S * q_dim * d         # o_proj
    attn_core = 2 * (2 * a.num_heads * S * S * hd)  # QK^T and softmax(.)V
    mlp = 3 * (2 * S * d * d_ff)           # gate, up, down (SwiGLU)
    lm_head = 2 * S * d * a.vocab_size

    return float(a.num_layers * (attn_proj + attn_core + mlp) + lm_head)


def _t5_encoder_flops(a: ArchSpec, S: int) -> float:
    d, h, hd, d_ff = a.hidden_size, a.num_heads, a.head_dim, a.intermediate_size
    q_dim = h * hd
    attn = 4 * (2 * S * d * q_dim) + 2 * (2 * h * S * S * hd)
    mlp = 3 * (2 * S * d * d_ff)           # gated-gelu: wi_0, wi_1, wo
    return float(a.num_layers * (attn + mlp))


def _t5_decoder_flops(a: ArchSpec, T: int, S: int) -> float:
    d, h, hd, d_ff = a.hidden_size, a.num_heads, a.head_dim, a.intermediate_size
    q_dim = h * hd
    self_attn = 4 * (2 * T * d * q_dim) + 2 * (2 * h * T * T * hd)
    cross_attn = 2 * (2 * T * d * q_dim)   # q, o over decoder tokens
    cross_attn += 2 * (2 * S * d * q_dim)  # k, v over encoder tokens
    cross_attn += 2 * (2 * h * T * S * hd)
    mlp = 3 * (2 * T * d * d_ff)
    lm_head = 2 * T * d * a.vocab_size
    return float(a.num_decoder_layers * (self_attn + cross_attn + mlp) + lm_head)


def _mamba1_forward_flops(a: ArchSpec, S: int) -> float:
    d, di, ds = a.hidden_size, a.d_inner, a.d_state
    in_proj = 2 * S * d * (2 * di)             # x and z
    conv = 2 * S * di * a.d_conv               # depthwise causal conv
    x_proj = 2 * S * di * (a.dt_rank + 2 * ds) # -> dt, B, C
    dt_proj = 2 * S * a.dt_rank * di
    scan = 9 * S * di * ds                     # selective scan recurrence (approx.)
    out_proj = 2 * S * di * d
    lm_head = 2 * S * d * a.vocab_size
    return float(a.num_layers * (in_proj + conv + x_proj + dt_proj + scan + out_proj) + lm_head)


def _mamba2_forward_flops(a: ArchSpec, S: int) -> float:
    d, di, ds, ng = a.hidden_size, a.d_inner, a.d_state, a.ngroups
    in_proj = 2 * S * d * (2 * di + 2 * ng * ds + a.num_heads)  # z, x, B, C, dt
    conv = 2 * S * (di + 2 * ng * ds) * a.d_conv
    scan = 9 * S * di * ds                     # SSD scan (approx.)
    out_proj = 2 * S * di * d
    lm_head = 2 * S * d * a.vocab_size
    return float(a.num_layers * (in_proj + conv + scan + out_proj) + lm_head)


def _causal_forward_flops(arch: ArchSpec, S: int) -> float:
    if arch.family == "transformer":
        return _transformer_forward_flops(arch, S)
    if arch.family == "mamba1":
        return _mamba1_forward_flops(arch, S)
    if arch.family == "mamba2":
        return _mamba2_forward_flops(arch, S)
    raise ValueError(f"Unknown architecture family: {arch.family}")


def training_flops(
    forward: Optional[float],
    method: Optional[str],
    tokens: int,
    trainable_params: Optional[int],
) -> Optional[float]:
    """Training (forward + backward) FLOPs for one sequence."""
    if forward is None or method is None:
        return None
    if method == "full_ft":
        return 3.0 * forward
    if method == "bitfit":
        return 2.0 * forward
    if trainable_params is None:
        return None
    adapter = 2.0 * tokens * trainable_params
    return 2.0 * forward + 3.0 * adapter


def estimate_flops(
    spec: ModelSpec,
    method: Optional[str],
    max_length: int,
    max_new_tokens: int,
    num_beams: int,
    trainable_params: Optional[int],
    total_params: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate per-sequence training and inference FLOPs for a run.

    Returns:
        Dict with ``seq_len``, ``max_new_tokens``, ``forward_flops_per_sequence``
        (training sequence), ``training_flops_per_sequence``,
        ``inference_flops_per_sequence`` and ``source``
        ("analytical", "approx_2NT" or "unavailable").
    """
    arch = spec.arch
    S, T = max_length, max_new_tokens
    beams = num_beams if spec.kind == "seq2seq" else 1
    train_tokens = S + T if spec.kind == "seq2seq" else S

    if arch is not None and arch.family == "t5":
        enc, dec = _t5_encoder_flops(arch, S), _t5_decoder_flops(arch, T, S)
        fwd_train, fwd_inf, source = enc + dec, enc + beams * dec, "analytical"
    elif arch is not None:
        fwd_train = _causal_forward_flops(arch, S)
        fwd_inf = _causal_forward_flops(arch, S + T)
        source = "analytical"
    elif total_params:
        fwd_train = 2.0 * total_params * train_tokens
        fwd_inf = 2.0 * total_params * (S + beams * T)
        source = "approx_2NT"
    else:
        fwd_train = fwd_inf = None
        source = "unavailable"

    return {
        "seq_len": S,
        "max_new_tokens": T,
        "num_beams": beams,
        "forward_flops_per_sequence": fwd_train,
        "training_flops_per_sequence": training_flops(fwd_train, method, train_tokens, trainable_params),
        "inference_flops_per_sequence": fwd_inf,
        "source": source,
    }


# =============================================================================
# NETSCORE
# =============================================================================

def calculate_netscore(
    a: Optional[float],
    p: Optional[float],
    m: Optional[float],
    v: Optional[float],
    t: Optional[float],
    w: Optional[float],
    beta: float,
    gamma: float,
    delta: float,
    lam: float,
    scale: float = 20.0,
    alpha: float = 2.0,
) -> Optional[float]:
    """
    Calculate a single NetScore value.

    Inputs must already be in NetScore units (see ``to_netscore_inputs``).
    A term whose exponent is 0 is ignored, so its value may be None.

    Returns:
        NetScore value, or None if performance is not positive or a term with a
        non-zero exponent is missing or not positive (log10 undefined).
    """
    if a is None or not a > 0:
        return None

    # Evaluated in log space: S * (alpha*log a - beta*log p - beta*log m - gamma*log v - delta*log t - lambda*log w)
    log_ratio = alpha * math.log10(a)
    for exponent, value in ((beta, p), (beta, m), (gamma, v), (delta, t), (lam, w)):
        if exponent == 0:
            continue
        if value is None or not value > 0:
            return None
        log_ratio -= exponent * math.log10(value)

    return scale * log_ratio


def to_netscore_inputs(
    performance: Optional[float],
    params: Optional[float],
    flops: Optional[float],
    vram_mb: Optional[float],
    time_s: Optional[float],
    power_w: Optional[float],
    cfg: NetScoreConfig = NETSCORE,
) -> Dict[str, Optional[float]]:
    """Convert raw measurements into NetScore units (keys a, p, m, v, t, w)."""
    def convert(value, unit):
        return None if value is None else float(value) / unit

    return {
        "a": None if performance is None else float(performance) * cfg.performance_scale,
        "p": convert(params, cfg.params_unit),
        "m": convert(flops, cfg.flops_unit),
        "v": convert(vram_mb, cfg.vram_mb_unit),
        "t": convert(time_s, cfg.time_s_unit),
        "w": convert(power_w, cfg.power_w_unit),
    }


def compute_netscore_variants(
    inputs: Dict[str, Optional[float]],
    cfg: NetScoreConfig = NETSCORE,
) -> Dict[str, Optional[float]]:
    """Compute every configured NetScore variant (NS, NS-E, NS-M, NS#)."""
    return {
        name: calculate_netscore(
            inputs["a"], inputs["p"], inputs["m"], inputs["v"], inputs["t"], inputs["w"],
            beta=exps["beta"],
            gamma=exps["gamma"],
            delta=exps["delta"],
            lam=exps["lambda"],
            scale=cfg.scale,
            alpha=cfg.alpha,
        )
        for name, exps in cfg.variants.items()
    }


def format_netscore_results(netscore: Dict[str, Optional[float]]) -> str:
    """Format NetScore variants for display."""
    lines = []
    for key, value in netscore.items():
        if value is not None:
            lines.append(f"   * {key}: {value:.4f}")
        else:
            lines.append(f"   * {key}: N/A (missing or non-positive inputs)")
    return "\n".join(lines)


def netscore_performance(task: str, metrics: Dict[str, Any], cfg: NetScoreConfig = NETSCORE) -> Optional[float]:
    """
    Map task metrics to the NetScore performance term in [0, 1] (higher is better).

    lamp1/lamp2: accuracy; lamp3: 1 - MAE / 4 (clipped at 0); lamp4: ROUGE-1.
    """
    kind = TASKS[task].kind
    if kind == "classification":
        return metrics.get("accuracy")
    if kind == "regression":
        mae = metrics.get("mae")
        if mae is None or not math.isfinite(mae):
            return None
        return max(0.0, 1.0 - mae / cfg.lamp3_max_error)
    return metrics.get("rouge1")


# =============================================================================
# GPU MONITORING: TRAINING CALLBACK AND INFERENCE TRACKER
# =============================================================================

class GPUReader:
    """Reads GPU power (W) and used VRAM (MiB) via NVML, falling back to nvidia-smi."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._handle = None
        if NVML_OK:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._handle = None

    def _smi(self, query: str) -> Optional[float]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--id={self.gpu_index}", f"--query-gpu={query}",
                 "--format=csv,noheader,nounits"],
                timeout=2,
            )
            return float(out.decode().strip())
        except Exception:
            return None

    def power_w(self) -> Optional[float]:
        if self._handle is not None:
            try:
                return pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            except Exception:
                return None
        return self._smi("power.draw")

    def vram_mb(self) -> Optional[float]:
        if self._handle is not None:
            try:
                return pynvml.nvmlDeviceGetMemoryInfo(self._handle).used / (1024 ** 2)
            except Exception:
                return None
        return self._smi("memory.used")


def _energy_wh_trapezoidal(samples: List[Dict[str, float]]) -> float:
    """Energy = integral of P dt (trapezoidal rule), in Wh."""
    if len(samples) < 2:
        return 0.0
    samples = sorted(samples, key=lambda s: s["timestamp"])
    energy_ws = 0.0
    for prev, cur in zip(samples, samples[1:]):
        energy_ws += 0.5 * (prev["power_w"] + cur["power_w"]) * (cur["timestamp"] - prev["timestamp"])
    return energy_ws / 3600.0


def _sample_stats(samples: List[Dict[str, float]]) -> Dict[str, Any]:
    powers = [s["power_w"] for s in samples]
    vrams = [s["vram_used_mb"] for s in samples]
    return {
        "num_samples": len(samples),
        "avg_power_watts": float(statistics.mean(powers)),
        "max_power_watts": float(max(powers)),
        "min_power_watts": float(min(powers)),
        "avg_vram_used_mb": float(statistics.mean(vrams)),
        "max_vram_used_mb": float(max(vrams)),
        "energy_wh": _energy_wh_trapezoidal(samples),
    }


class NVMLTrainingCallback(TrainerCallback):
    """
    Samples GPU power and VRAM during training.

    Two sources, the first one with data is used for energy / power / VRAM:
      1. Continuous sampling: a background thread every ``sample_interval_ms``.
      2. Step sampling: one reading every ``sample_every_n_steps`` optimizer steps.
    Energy is integrated with the trapezoidal rule and reported in Wh.
    """

    def __init__(
        self,
        output_dir: str,
        gpu_index: int = 0,
        sample_interval_ms: int = 100,
        sample_every_n_steps: int = 10,
        use_background_sampling: bool = True,
    ):
        self.output_dir = output_dir
        self.reader = GPUReader(gpu_index)
        self.sample_interval_ms = sample_interval_ms
        self.sample_every_n_steps = max(1, sample_every_n_steps)
        self.use_background_sampling = use_background_sampling

        self.continuous_samples: List[Dict[str, float]] = []
        self.step_samples: List[Dict[str, float]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.train_start_time: Optional[float] = None
        self.summary: Dict[str, Any] = {}

    def _read(self) -> Optional[Dict[str, float]]:
        power, vram = self.reader.power_w(), self.reader.vram_mb()
        if power is None or vram is None:
            return None
        return {"power_w": power, "vram_used_mb": vram, "timestamp": time.time()}

    def _sampling_loop(self):
        interval = self.sample_interval_ms / 1000.0
        while self._running:
            sample = self._read()
            if sample:
                with self._lock:
                    self.continuous_samples.append(sample)
            time.sleep(interval)

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        self.continuous_samples, self.step_samples = [], []
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        if self.use_background_sampling:
            self._running = True
            self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
            self._thread.start()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.sample_every_n_steps == 0:
            sample = self._read()
            if sample:
                sample["global_step"] = state.global_step
                self.step_samples.append(sample)

    def on_train_end(self, args, state, control, **kwargs):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            continuous = list(self.continuous_samples)

        summary: Dict[str, Any] = {
            "training_duration_seconds": time.time() - self.train_start_time if self.train_start_time else None,
        }
        if continuous:
            summary["continuous_sampling"] = {"sample_interval_ms": self.sample_interval_ms,
                                              **_sample_stats(continuous)}
        if self.step_samples:
            summary["step_sampling"] = {"sample_every_n_steps": self.sample_every_n_steps,
                                        **_sample_stats(self.step_samples)}
        best = "continuous_sampling" if continuous else ("step_sampling" if self.step_samples else None)
        if best:
            summary.update({
                "energy_source": best,
                "estimated_energy_wh": summary[best]["energy_wh"],
                "avg_power_watts": summary[best]["avg_power_watts"],
                "max_power_watts": summary[best]["max_power_watts"],
                "avg_vram_used_mb": summary[best]["avg_vram_used_mb"],
                "max_vram_used_mb": summary[best]["max_vram_used_mb"],
                "num_power_samples": summary[best]["num_samples"],
            })
        if torch.cuda.is_available():
            try:
                summary["peak_torch_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            except Exception:
                pass
        self.summary = summary

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "power_vram_timeseries.json"), "w") as f:
                json.dump({"summary": summary, "step_samples": self.step_samples}, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to save NVML timeseries: {e}")


class GPUInferenceTracker:
    """Tracks per-sample and aggregate inference metrics: time, power, energy, VRAM.

    Separates **total time** (retrieval + tokenization + generation + decoding)
    from **generation time** (model.generate only), and records generated
    token counts for throughput. A lightweight background thread polls GPU
    power/memory while each sample is processed.
    """

    def __init__(self, gpu_index: int = 0, poll_interval: float = 0.05):
        self.reader = GPUReader(gpu_index)
        self.poll_interval = poll_interval

        self._power: List[float] = []
        self._vram: List[float] = []
        self._sample_start = 0.0
        self._gen_start = 0.0
        self._polling = False
        self._thread: Optional[threading.Thread] = None

        self.sample_total_times: List[float] = []
        self.sample_gen_times: List[float] = []
        self.sample_num_tokens: List[int] = []
        self.sample_energy: List[float] = []
        self.sample_avg_power: List[float] = []
        self.sample_peak_vram: List[float] = []

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _poll_loop(self):
        while self._polling:
            pw, vr = self.reader.power_w(), self.reader.vram_mb()
            if pw is not None:
                self._power.append(pw)
            if vr is not None:
                self._vram.append(vr)
            time.sleep(self.poll_interval)

    def start_sample(self):
        """Call at the very beginning of a sample (before retrieval/tokenization)."""
        self._power, self._vram = [], []
        self._sync()
        self._sample_start = time.perf_counter()
        self._polling = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def start_generation(self):
        """Call right before model.generate()."""
        self._sync()
        self._gen_start = time.perf_counter()

    def end_generation(self, num_new_tokens: int = 0):
        """Call right after model.generate()."""
        self._sync()
        self.sample_gen_times.append(time.perf_counter() - self._gen_start)
        self.sample_num_tokens.append(int(num_new_tokens))

    def end_sample(self):
        """Call at the very end of a sample (after decoding)."""
        self._sync()
        elapsed = time.perf_counter() - self._sample_start
        self._polling = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        self.sample_total_times.append(elapsed)
        avg_pw = sum(self._power) / len(self._power) if self._power else 0.0
        self.sample_avg_power.append(avg_pw)
        self.sample_energy.append(avg_pw * elapsed)
        self.sample_peak_vram.append(max(self._vram) if self._vram else 0.0)

    def summary(self) -> Dict[str, Any]:
        """Return aggregate inference statistics (with per-sample detail)."""
        n = len(self.sample_total_times)
        if n == 0:
            return {}
        total_time = sum(self.sample_total_times)
        total_gen = sum(self.sample_gen_times)
        total_tokens = sum(self.sample_num_tokens)
        total_energy = sum(self.sample_energy)
        has_power = any(p > 0 for p in self.sample_avg_power)
        has_vram = any(v > 0 for v in self.sample_peak_vram)

        return {
            "num_samples": n,
            # End-to-end (retrieval + tokenization + generation + decoding)
            "total_time_s": round(total_time, 3),
            "avg_time_per_sample_s": round(total_time / n, 4),
            # Generation only (model.generate)
            "total_generation_time_s": round(total_gen, 3),
            "avg_generation_time_per_sample_s": round(total_gen / n, 4),
            "inference_latency_s": round(total_gen / n, 4),
            # Token throughput
            "total_tokens_generated": total_tokens,
            "throughput_tokens_per_s": round(total_tokens / total_gen, 2) if total_gen > 0 else 0.0,
            # Energy / power (None when no GPU readings were available)
            "total_energy_j": round(total_energy, 3) if has_power else None,
            "total_energy_wh": round(total_energy / 3600.0, 6) if has_power else None,
            "avg_energy_per_sample_j": round(total_energy / n, 4) if has_power else None,
            "avg_power_w": round(sum(self.sample_avg_power) / n, 2) if has_power else None,
            # VRAM
            "peak_vram_mb": round(max(self.sample_peak_vram), 2) if has_vram else None,
            "avg_peak_vram_per_sample_mb": round(sum(self.sample_peak_vram) / n, 2) if has_vram else None,
            # Per-sample detail
            "per_sample_total_time_s": [round(t, 4) for t in self.sample_total_times],
            "per_sample_generation_time_s": [round(t, 4) for t in self.sample_gen_times],
            "per_sample_num_tokens": list(self.sample_num_tokens),
            "per_sample_energy_j": [round(e, 4) for e in self.sample_energy],
            "per_sample_avg_power_w": [round(p, 2) for p in self.sample_avg_power],
            "per_sample_peak_vram_mb": [round(v, 2) for v in self.sample_peak_vram],
        }


def strip_per_sample(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Drop per-sample lists (kept in inference_stats.json) for the main results file."""
    return {k: v for k, v in (stats or {}).items() if not k.startswith("per_sample_")}


# =============================================================================
# MAMBA2 WRAPPER (mamba_ssm backbone made HuggingFace/PEFT compatible)
# =============================================================================

class Mamba2ConfigWrapper(PretrainedConfig):
    """Wrapper to make a mamba_ssm config compatible with HuggingFace/PEFT."""
    model_type = "mamba2"

    def __init__(self, mamba_config=None, **kwargs):
        super().__init__(**kwargs)
        if mamba_config is not None:
            for key, value in vars(mamba_config).items():
                setattr(self, key, value)
        self.hidden_size = getattr(self, "d_model", 2048)
        self.num_hidden_layers = getattr(self, "n_layer", 64)


class Mamba2ForCausalLM(PreTrainedModel):
    """Causal-LM wrapper around mamba_ssm's MambaLMHeadModel for Trainer/PEFT."""
    config_class = Mamba2ConfigWrapper
    base_model_prefix = "backbone"
    _tied_weights_keys = ["backbone.lm_head.weight"]

    def __init__(self, config: Mamba2ConfigWrapper, backbone=None):
        super().__init__(config)
        self.backbone = backbone
        self.config = config

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        lm_logits = self.backbone(input_ids).logits  # (batch, seq_len, vocab)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return CausalLMOutput(loss=loss, logits=lm_logits, hidden_states=None)

    def get_input_embeddings(self):
        return self.backbone.backbone.embedding

    def set_input_embeddings(self, value):
        self.backbone.backbone.embedding = value

    def generate(self, input_ids, max_new_tokens=32, **kwargs):
        return self.backbone.generate(
            input_ids=input_ids, max_length=input_ids.shape[1] + max_new_tokens, **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


def load_mamba2(source: str, device: torch.device, dtype: torch.dtype) -> Mamba2ForCausalLM:
    """Load a Mamba2 model from a Hub id or a local checkpoint saved by this script."""
    if not MAMBA_SSM_OK:
        raise ImportError(
            "mamba_ssm is required for mamba2. Install with:\n"
            "  pip install causal-conv1d>=1.2.0 mamba-ssm --no-build-isolation"
        )
    config_data = load_config_hf(source)
    valid_keys = {f.name for f in dataclass_fields(_MambaConfig)}
    cfg = _MambaConfig(**{k: v for k, v in config_data.items() if k in valid_keys})
    backbone = MambaLMHeadModel(cfg, device=device, dtype=dtype)
    state_dict = load_state_dict_hf(source, device=device, dtype=dtype)
    # Checkpoints saved through the wrapper carry an extra "backbone." prefix
    prefix = "backbone."
    if any(k.startswith(prefix + "backbone.") or k == prefix + "lm_head.weight" for k in state_dict):
        state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    backbone.load_state_dict(state_dict)
    print(f"   Mamba2 config: d_model={backbone.config.d_model}, n_layer={backbone.config.n_layer}")
    return Mamba2ForCausalLM(Mamba2ConfigWrapper(backbone.config), backbone=backbone)


# =============================================================================
# CONTRIEVER RETRIEVAL
# =============================================================================

def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling for sentence embeddings."""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


class ContrieverRetriever:
    """Contriever-based retrieval of the most relevant user-profile items."""

    def __init__(self, device: str = "cuda:0", checkpoint: str = "facebook/contriever"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(device)
        self.model.eval()

    @torch.no_grad()
    def retrieve_top_k(self, corpus: List[str], profile: List[Dict], query: str,
                       k: int, batch_size: int = 4) -> List[Dict]:
        """Retrieve the top-k profile items by query/corpus similarity."""
        if not profile:
            return []
        k = min(k, len(profile))

        q = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(self.device)
        q_emb = mean_pooling(self.model(**q).last_hidden_state, q["attention_mask"])

        scores: List[float] = []
        for i in range(0, len(corpus), batch_size):
            batch = self.tokenizer(corpus[i:i + batch_size], padding=True, truncation=True,
                                   return_tensors="pt").to(self.device)
            emb = mean_pooling(self.model(**batch).last_hidden_state, batch["attention_mask"])
            s = q_emb.squeeze() @ emb.T
            scores.extend([s.item()] if s.dim() == 0 else s.tolist())

        _, top = torch.topk(torch.tensor(scores), k)
        return [profile[m] for m in top.tolist()]

    def to_cpu(self):
        """Move the retriever to CPU to free GPU memory during training."""
        self.model = self.model.to("cpu")
        self.device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_device(self, device: str):
        """Move the retriever (back) to ``device``."""
        self.model = self.model.to(device)
        self.device = device


# =============================================================================
# QUERY / CORPUS MAKERS AND PROMPT CREATORS
# =============================================================================

def extract_strings_between_quotes(input_string: str) -> List[str]:
    """Extract strings between double quotes."""
    output, inside, current = [], False, ""
    for char in input_string:
        if char == '"' and not inside:
            inside = True
        elif char == '"' and inside:
            inside = False
            output.append(current)
            current = ""
        elif inside:
            current += char
    return output


def extract_after_keyword(input_string: str, keyword: str) -> str:
    """Extract text after a keyword (whole string if the keyword is absent)."""
    index = input_string.find(keyword)
    if index == -1:
        return input_string
    return input_string[index + len(keyword):].strip()


def lamp1_query_corpus(inp: str, profile: List[Dict]) -> Tuple[List[str], str]:
    """LaMP-1: citation identification. Profile: title, abstract, id."""
    corpus = [f'{x["title"]} {x.get("abstract", "")}' for x in profile]
    extracted = extract_strings_between_quotes(inp)
    query = f"{extracted[1]} {extracted[2]}" if len(extracted) >= 3 else inp
    return corpus, query


def lamp2_query_corpus(inp: str, profile: List[Dict]) -> Tuple[List[str], str]:
    """LaMP-2: movie tagging. Profile: description, tag, id."""
    return [f'{x.get("description", "")}' for x in profile], extract_after_keyword(inp, "description:")


def lamp3_query_corpus(inp: str, profile: List[Dict]) -> Tuple[List[str], str]:
    """LaMP-3: product rating. Profile: text, score, id."""
    return [f'{x.get("text", "")}' for x in profile], extract_after_keyword(inp, "review:")


def lamp4_query_corpus(inp: str, profile: List[Dict]) -> Tuple[List[str], str]:
    """LaMP-4: news headline generation. Profile: title, text, id."""
    return ([f'{x.get("title", "")} {x.get("text", "")}' for x in profile],
            extract_after_keyword(inp, "article:"))


QUERY_CORPUS_MAKERS = {
    "lamp1": lamp1_query_corpus,
    "lamp2": lamp2_query_corpus,
    "lamp3": lamp3_query_corpus,
    "lamp4": lamp4_query_corpus,
}


def add_string_after_title(original_string: str, string_to_add: str) -> str:
    """Insert context after the 'title' keyword."""
    idx = original_string.find("title")
    if idx == -1:
        return string_to_add + " " + original_string
    return original_string[:idx + 5] + ", and " + string_to_add + original_string[idx + 5:]


def create_lamp1_prompt(inp: str, profile: List[Dict], max_length: int, tokenizer) -> str:
    """LaMP-1 prompt: inserts retrieved paper titles after 'title'."""
    if not profile:
        return inp
    prompts = []
    per_p_max_length = max((max_length - 2 * (len(profile) - 1)) // len(profile), 10)
    saved_tokens = 0
    for p in profile:
        tokens = tokenizer(p["title"], max_length=per_p_max_length + saved_tokens - 2, truncation=True)
        saved_tokens += per_p_max_length - len(tokens["input_ids"]) - 2
        new_title = tokenizer.batch_decode([tokens["input_ids"]], skip_special_tokens=True)[0]
        prompts.append(f'"{new_title}"')
    return add_string_after_title(inp, ", and ".join(prompts))


def _ppep_prompt(inp: str, profile: List[Dict], max_length: int, tokenizer,
                 text_key: str, template) -> str:
    """Shared PPEP/AIP builder: concat(template(item), ', and '). [INPUT]"""
    if not profile:
        return inp
    per_p_max_length = max((max_length - 1 - 2 * (len(profile) - 1)) // len(profile), 10)
    saved_tokens = 0
    prompts = []
    for p in profile:
        needed_part_len = len(tokenizer(template(p, " "))["input_ids"])
        tokens = tokenizer(p.get(text_key, ""), max_length=per_p_max_length + saved_tokens - needed_part_len,
                           truncation=True)
        saved_tokens += per_p_max_length - len(tokens["input_ids"]) - needed_part_len
        new_text = tokenizer.batch_decode([tokens["input_ids"]], skip_special_tokens=True)[0]
        prompts.append(template(p, new_text))
    return f'{", and ".join(prompts)}. {inp}'


def create_lamp2_prompt(inp, profile, max_length, tokenizer) -> str:
    """LaMP-2 PPEP: the tag for the movie: "[description]" is "[tag]"."""
    return _ppep_prompt(inp, profile, max_length, tokenizer, "description",
                        lambda p, text: f'the tag for the movie: "{text}" is "{p.get("tag", "")}"')


def create_lamp3_prompt(inp, profile, max_length, tokenizer) -> str:
    """LaMP-3 PPEP: [score] is the score for "[text]"."""
    return _ppep_prompt(inp, profile, max_length, tokenizer, "text",
                        lambda p, text: f'{p.get("score", "")} is the score for "{text}"')


def create_lamp4_prompt(inp, profile, max_length, tokenizer) -> str:
    """LaMP-4 PPEP: "[title]" is the title for "[text]"."""
    return _ppep_prompt(inp, profile, max_length, tokenizer, "text",
                        lambda p, text: f'"{p.get("title", "")}" is the title for "{text}"')


PROMPT_CREATORS = {
    "lamp1": create_lamp1_prompt,
    "lamp2": create_lamp2_prompt,
    "lamp3": create_lamp3_prompt,
    "lamp4": create_lamp4_prompt,
}


def create_augmented_prompt(item: Dict, task: str, tokenizer, retriever: Optional[ContrieverRetriever],
                            num_retrieved: int, max_length: int, max_profile_size: int) -> str:
    """Build the profile-augmented input (identical for training and evaluation)."""
    inp = item["input"]
    profile = item.get("profile", [])[:max_profile_size]
    if not profile:
        return inp

    if retriever is not None:
        corpus, query = QUERY_CORPUS_MAKERS[task](inp, profile)
        selected = retriever.retrieve_top_k(corpus, profile, query, num_retrieved)
    else:
        selected = profile[:num_retrieved]

    # Shrink the input budget until the profile prompt fits
    factor = 0.6
    while factor > 0:
        try:
            max_len_prompt = max_length - min(len(tokenizer(inp)["input_ids"]), int(factor * max_length))
            return PROMPT_CREATORS[task](inp, selected, max_len_prompt, tokenizer)
        except Exception:
            factor -= 0.1
    return inp


# =============================================================================
# DATA
# =============================================================================

def load_lamp_split(split_dir: str) -> List[Dict]:
    """
    Load one LaMP split:
        split_dir/inputs.json  - list of {id, input, profile}
        split_dir/outputs.json - list of {id, output} or {"golds": [...]}
    """
    with open(os.path.join(split_dir, "inputs.json"), "r", encoding="utf-8") as f:
        inputs_data = json.load(f)
    with open(os.path.join(split_dir, "outputs.json"), "r", encoding="utf-8") as f:
        outputs_data = json.load(f)
    if isinstance(outputs_data, dict) and "golds" in outputs_data:
        outputs_data = outputs_data["golds"]

    output_lookup = {item["id"]: item["output"] for item in outputs_data}
    return [
        {"id": item["id"], "input": item["input"], "profile": item.get("profile", []),
         "output": output_lookup[item["id"]]}
        for item in inputs_data if item["id"] in output_lookup
    ]


def resolve_task_data_dir(data_dir: str, task: str, multi_task: bool) -> str:
    """data_dir is the dataset folder for one task, or the parent of datasetN/ folders."""
    nested = os.path.join(data_dir, TASKS[task].dataset_folder)
    if multi_task:
        return nested
    if not os.path.isdir(os.path.join(data_dir, "validation")) and os.path.isdir(nested):
        return nested
    return data_dir


# Prompt templates (kept identical to the original scripts for comparability)
def build_generation_prompt(model_type: str, source: str) -> str:
    if model_type in ("mamba1", "mamba2"):
        return f"{source}\nAnswer:"
    if model_type == "qwen3":
        return f"<|im_start|>user\n{source}<|im_end|>\n<|im_start|>assistant\n"
    if model_type == "tinyllama":
        return f"<s>[INST] {source} [/INST]"
    return source  # flan-t5


def build_training_text(model_type: str, source: str, target: str) -> str:
    if model_type in ("mamba1", "mamba2"):
        return f"{source}\nAnswer: {target}"
    if model_type == "qwen3":
        return f"<|im_start|>user\n{source}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>"
    return f"<s>[INST] {source} [/INST] {target}</s>"  # tinyllama


def build_hf_dataset(data: List[Dict], task: str, tokenizer, retriever, opts, desc: str) -> datasets.Dataset:
    """Retrieve profiles, build prompts, and return a HF Dataset of {id, source, target}."""
    records = [
        {"id": str(item["id"]),
         "source": create_augmented_prompt(item, task, tokenizer, retriever, opts.num_retrieved,
                                           opts.max_length, opts.max_profile_size),
         "target": item["output"]}
        for item in tqdm(data, desc=desc)
    ]
    return datasets.Dataset.from_list(records)


def make_preprocessor(spec: ModelSpec, model_type: str, tokenizer, max_length: int):
    """Tokenization for seq2seq (source -> target) or causal LM (template, labels = input_ids)."""
    if spec.kind == "seq2seq":
        def preprocess(examples):
            return tokenizer(examples["source"], text_target=examples["target"],
                             max_length=max_length, truncation=True)
        return preprocess

    def preprocess(examples):
        texts = [build_training_text(model_type, s, t) for s, t in zip(examples["source"], examples["target"])]
        enc = tokenizer(texts, max_length=max_length, truncation=True, padding=False)
        enc["labels"] = [list(ids) for ids in enc["input_ids"]]
        return enc
    return preprocess


# =============================================================================
# MODEL LOADING AND METHOD SETUP
# =============================================================================

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(opts, training: bool) -> torch.dtype:
    """--bf16 / --fp16 select the weight dtype; otherwise fp32 for training, bf16/fp16 for evaluation."""
    if opts.bf16:
        return torch.bfloat16
    if opts.fp16:
        return torch.float16
    if training or not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_bnb_config(opts, dtype: torch.dtype) -> BitsAndBytesConfig:
    """NF4 (or 8-bit) quantization config for QLoRA."""
    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError("QLoRA requires bitsandbytes. Install with: pip install bitsandbytes")
    compute_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    print(f"   QLoRA: {opts.qlora_bits}-bit quantization, compute_dtype={compute_dtype}")
    return BitsAndBytesConfig(
        load_in_4bit=(opts.qlora_bits == 4),
        load_in_8bit=(opts.qlora_bits == 8),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(spec: ModelSpec, model_type: str, model_name: str, cache_dir: str):
    """Load the tokenizer (causal models get a pad token; non-Llama causal models pad left)."""
    name = spec.tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, trust_remote_code=True)
    if spec.kind == "causal":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model_type != "tinyllama":
            tokenizer.padding_side = "left"  # Required for Mamba2; used for generation
    print(f"   Tokenizer: {name} (padding_side={tokenizer.padding_side})")
    return tokenizer


def load_model(spec: ModelSpec, model_type: str, source: str, tokenizer, dtype: torch.dtype,
               bnb_config: Optional[BitsAndBytesConfig], device_map: Optional[str], cache_dir: str):
    """Load a base model or a full checkpoint from ``source``."""
    print(f"   Loading {model_type} weights from {source} (dtype={dtype})")
    if model_type == "mamba2":
        return load_mamba2(source, get_device(), dtype)

    kwargs: Dict[str, Any] = dict(cache_dir=cache_dir, torch_dtype=dtype, trust_remote_code=True)
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
    if device_map is not None:
        kwargs["device_map"] = device_map
    cls = AutoModelForSeq2SeqLM if spec.kind == "seq2seq" else AutoModelForCausalLM
    model = cls.from_pretrained(source, **kwargs)

    if model_type == "mamba1" and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if spec.kind == "causal":
        model.config.pad_token_id = tokenizer.pad_token_id
    return model


def bitfit_param_names(model) -> List[str]:
    """Bias parameters; models without biases (LLaMA, Qwen3, T5) fall back to norm weights."""
    names = [n for n, _ in model.named_parameters() if "bias" in n]
    if not names:
        names = [n for n, _ in model.named_parameters() if "norm" in n.lower() and "weight" in n.lower()]
    return names


def count_parameters(model) -> Tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def adapted_parameter_count(model, method: Optional[str]) -> Optional[int]:
    """Parameters updated by ``method`` (for evaluation, where nothing requires grad)."""
    if method is None:
        return None  # zero-shot: nothing was fine-tuned -> NS (which needs p) is null
    if method in LORA_METHODS:
        return int(sum(p.numel() for n, p in model.named_parameters() if "lora_" in n))
    if method == "bitfit":
        names = set(bitfit_param_names(model))
        return int(sum(p.numel() for n, p in model.named_parameters() if n in names))
    return count_parameters(model)[1]


def apply_method(model, spec: ModelSpec, method: str, opts):
    """Apply LoRA / LoRA+ / QLoRA adapters, BitFit freezing, or full fine-tuning."""
    if method == "full_ft":
        print("   Full fine-tuning: all parameters are trainable")
        for p in model.parameters():
            p.requires_grad = True
    elif method == "bitfit":
        names = set(bitfit_param_names(model))
        kind = "bias" if any("bias" in n for n in names) else "norm-weight (no biases found)"
        print(f"   BitFit: training {kind} parameters")
        for n, p in model.named_parameters():
            p.requires_grad = n in names
    else:
        if method == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        lora_config = LoraConfig(
            r=opts.lora_r,
            lora_alpha=opts.lora_alpha,
            lora_dropout=opts.lora_dropout,
            target_modules=spec.lora_targets,
            task_type=spec.lora_task_type,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if method == "loraplus":
            print(f"   LoRA+ ratio: {opts.loraplus_ratio} (lr_B = lr * {opts.loraplus_ratio})")

    trainable, total = count_parameters(model)
    print(f"   trainable params: {trainable:,} || all params: {total:,} || "
          f"trainable%: {100.0 * trainable / max(total, 1):.4f}")
    return model


def build_loraplus_optimizer(model, lr: float, loraplus_ratio: int, weight_decay: float = 0.01):
    """
    LoRA+ optimizer: lr for A matrices, lr * ratio for B matrices.

    Uses PEFT's create_loraplus_optimizer (Adam8bit if bitsandbytes is present),
    falling back to manual parameter groups.
    """
    try:
        from peft.optimizers import create_loraplus_optimizer
        try:
            import bitsandbytes as bnb
            opt_cls = bnb.optim.Adam8bit
        except ImportError:
            opt_cls = torch.optim.AdamW
        optimizer = create_loraplus_optimizer(
            model=model, optimizer_cls=opt_cls, lr=lr,
            loraplus_lr_ratio=loraplus_ratio, weight_decay=weight_decay,
        )
        print(f"   Using PEFT create_loraplus_optimizer ({opt_cls.__name__}, ratio={loraplus_ratio})")
        return optimizer
    except Exception as e:
        print(f"   PEFT LoRA+ optimizer not available ({e}), using manual param groups")

    a_params, b_params, rest = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (a_params if "lora_A" in name else b_params if "lora_B" in name else rest).append(param)
    groups = [{"params": a_params, "lr": lr}, {"params": b_params, "lr": lr * loraplus_ratio}]
    if rest:
        groups.append({"params": rest, "lr": lr})
    try:
        import bitsandbytes as bnb
        return bnb.optim.Adam8bit(groups, weight_decay=weight_decay)
    except ImportError:
        return torch.optim.AdamW(groups, weight_decay=weight_decay)


class LoraPlusTrainer(Trainer):
    """Trainer that builds the LoRA+ optimizer; the scheduler is created by Trainer
    with the correct number of steps (accounts for gradient accumulation)."""

    def __init__(self, *args, loraplus_ratio: int = 16, **kwargs):
        self.loraplus_ratio = loraplus_ratio
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = build_loraplus_optimizer(
                self.model, self.args.learning_rate, self.loraplus_ratio, self.args.weight_decay
            )
        return self.optimizer


# =============================================================================
# GENERATION-BASED EVALUATION AND METRICS
# =============================================================================

@torch.no_grad()
def generate_predictions(model, tokenizer, data: List[Dict], task: str, model_type: str,
                         retriever: Optional[ContrieverRetriever], opts,
                         tracker: Optional[GPUInferenceTracker] = None) -> List[Dict]:
    """Generate one prediction per sample (batch size 1), optionally tracking GPU cost."""
    spec = MODELS[model_type]
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    for item in tqdm(data, desc=f"Generating ({model_type})"):
        if tracker:
            tracker.start_sample()

        source = create_augmented_prompt(item, task, tokenizer, retriever, opts.num_retrieved,
                                         opts.max_length, opts.max_profile_size)
        prompt = build_generation_prompt(model_type, source)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=opts.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if tracker:
            tracker.start_generation()
        if model_type == "mamba2":
            outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=opts.max_new_tokens)
        elif model_type == "mamba1":  # SSM: no attention_mask
            outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=opts.max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
        elif spec.kind == "seq2seq":
            outputs = model.generate(**inputs, max_new_tokens=opts.max_new_tokens, do_sample=False,
                                     num_beams=opts.num_beams)
        else:
            outputs = model.generate(**inputs, max_new_tokens=opts.max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        generated = outputs[0] if spec.kind == "seq2seq" else outputs[0][inputs["input_ids"].shape[1]:]
        if tracker:
            tracker.end_generation(num_new_tokens=len(generated))

        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if tracker:
            tracker.end_sample()

        predictions.append({"id": item["id"], "prediction": prediction, "ground_truth": item["output"]})

    return predictions


def _parse_score(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        return float(numbers[0]) if numbers else None


def compute_metrics_classification(predictions: List[Dict]) -> Dict[str, Any]:
    """Accuracy and macro-F1 for lamp1/lamp2 (exact match, then partial label match)."""
    preds = [p["prediction"].strip().lower() for p in predictions]
    golds = [p["ground_truth"].strip().lower() for p in predictions]
    unique_labels = sorted(set(golds))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    pred_indices, gold_indices = [], []
    for pred, gold in zip(preds, golds):
        pred_idx = label_to_idx.get(pred, -1)
        if pred_idx == -1 and pred:  # empty predictions never match
            for label, idx in label_to_idx.items():
                if label in pred or pred in label:
                    pred_idx = idx
                    break
        pred_indices.append(pred_idx)
        gold_indices.append(label_to_idx.get(gold, -1))

    total = len(predictions)
    correct = sum(1 for p, g in zip(pred_indices, gold_indices) if p == g and p != -1)
    valid_pairs = [(p, g) for p, g in zip(pred_indices, gold_indices) if p != -1 and g != -1]
    f1 = 0.0
    if valid_pairs:
        valid_preds, valid_golds = zip(*valid_pairs)
        try:
            f1 = float(f1_score(valid_golds, valid_preds, labels=sorted(set(valid_golds)), average="macro"))
        except Exception:
            f1 = 0.0

    print("\nSample predictions:")
    for i, p in enumerate(predictions[:10]):
        match = "OK " if pred_indices[i] == gold_indices[i] and pred_indices[i] != -1 else "ERR"
        print(f"  [{match}] Pred: '{p['prediction'][:50]}' | Gold: '{p['ground_truth'][:50]}'")

    return {
        "accuracy": correct / total if total else 0.0,
        "f1": f1,
        "correct": correct,
        "total_samples": total,
        "valid_predictions": sum(1 for p in pred_indices if p != -1),
        "unique_labels": unique_labels,
    }


def compute_metrics_regression(predictions: List[Dict]) -> Dict[str, Any]:
    """MAE and RMSE for lamp3 (predictions clamped to the 1-5 rating range)."""
    pred_scores, gold_scores, invalid = [], [], 0
    for p in predictions:
        gold = _parse_score(p["ground_truth"].strip())
        if gold is None:
            continue
        pred = _parse_score(p["prediction"].strip())
        if pred is None:
            invalid += 1
            continue
        pred_scores.append(max(1.0, min(5.0, pred)))
        gold_scores.append(gold)

    print("\nSample predictions:")
    for p in predictions[:10]:
        print(f"  Pred: '{p['prediction'][:20]}' | Gold: '{p['ground_truth']}'")

    if not pred_scores:
        print("\nWarning: no valid numeric predictions found!")
        return {"mae": float("inf"), "rmse": float("inf"), "total_samples": len(predictions),
                "valid_predictions": 0, "invalid_predictions": invalid}
    return {
        "mae": float(mean_absolute_error(gold_scores, pred_scores)),
        "rmse": float(math.sqrt(mean_squared_error(gold_scores, pred_scores))),
        "total_samples": len(predictions),
        "valid_predictions": len(pred_scores),
        "invalid_predictions": invalid,
    }


def compute_metrics_rouge(predictions: List[Dict]) -> Dict[str, Any]:
    """ROUGE-1 and ROUGE-L F-measure for lamp4."""
    if not ROUGE_AVAILABLE:
        raise ImportError("lamp4 requires rouge-score. Install with: pip install rouge-score")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1, rl = [], []
    for p in predictions:
        pred, gold = p["prediction"].strip(), p["ground_truth"].strip()
        if not pred:
            r1.append(0.0)
            rl.append(0.0)
            continue
        s = scorer.score(gold, pred)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    print("\nSample predictions:")
    for i, p in enumerate(predictions[:10]):
        print(f"  R1:{r1[i]:.2f} | Pred: '{p['prediction'][:50]}' | Gold: '{p['ground_truth'][:50]}'")

    return {
        "rouge1": float(np.mean(r1)) if r1 else 0.0,
        "rougeL": float(np.mean(rl)) if rl else 0.0,
        "total_samples": len(predictions),
        "valid_predictions": sum(1 for s in r1 if s > 0),
    }


def compute_task_metrics(predictions: List[Dict], task: str) -> Dict[str, Any]:
    kind = TASKS[task].kind
    if kind == "classification":
        return compute_metrics_classification(predictions)
    if kind == "regression":
        return compute_metrics_regression(predictions)
    return compute_metrics_rouge(predictions)


# =============================================================================
# RUNNERS
# =============================================================================

def _save_json(path: str, data: Any):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"   Saved: {path}")


def _free_training_state(trainer, model):
    """Free optimizer/scheduler/grad memory before the inference pass."""
    if trainer is not None:
        trainer.optimizer = None
        trainer.lr_scheduler = None
        try:
            trainer.accelerator.free_memory()
        except Exception:
            pass
    for p in model.parameters():
        p.grad = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _check_supported(model_type: str, method: Optional[str]):
    if model_type == "mamba2" and not MAMBA_SSM_OK:
        raise ImportError("mamba_ssm is required for mamba2 (pip install causal-conv1d>=1.2.0 mamba-ssm "
                          "--no-build-isolation)")
    if model_type == "mamba2" and method == "qlora":
        raise ValueError("QLoRA is not supported with mamba2 (mamba_ssm kernels do not work with "
                         "bitsandbytes). Use lora, loraplus, or full_ft instead.")


def _load_validation(data_dir: str, opts, limit: Optional[int] = None) -> List[Dict]:
    val_data = load_lamp_split(os.path.join(data_dir, "validation"))
    limit = limit or opts.max_samples
    if limit:
        val_data = val_data[:limit]
    print(f"   Validation samples: {len(val_data)}")
    return val_data


def _print_metrics(title: str, metrics: Dict[str, Any]):
    print(f"\n{title}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   * {key}: {value:.4f}")
        elif not isinstance(value, list):
            print(f"   * {key}: {value}")


def _print_inference_stats(stats: Dict[str, Any]):
    if not stats:
        return
    fmt = lambda v, spec: "N/A" if v is None else format(v, spec)
    print("\nInference metrics:")
    print(f"   * Latency (avg generate/sample): {stats['inference_latency_s']:.4f} s")
    print(f"   * Total end-to-end time:         {stats['total_time_s']:.3f} s")
    print(f"   * Throughput:                    {stats['throughput_tokens_per_s']:.2f} tokens/s")
    print(f"   * Total energy:                  {fmt(stats['total_energy_wh'], '.6f')} Wh")
    print(f"   * Avg power:                     {fmt(stats['avg_power_w'], '.2f')} W")
    print(f"   * Peak VRAM:                     {fmt(stats['peak_vram_mb'], '.2f')} MiB")


def run_training(opts, model_type: str, task: str, method: str, data_dir: str, run_dir: str,
                 retriever: Optional[ContrieverRetriever]) -> Dict[str, Any]:
    """Fine-tune, evaluate by generation (with GPU tracking), compute FLOPs and NetScore."""
    spec = MODELS[model_type]
    task_cfg = TASKS[task]
    model_name = opts.model_name or spec.hf_name
    _check_supported(model_type, method)

    print(f"\n{'=' * 70}")
    print(f"{METHOD_NAMES[method]} fine-tuning | model={model_name} | task={task} ({task_cfg.description})")
    print(f"Data: {data_dir}\nOutput: {run_dir}\nNVML available: {NVML_OK}")
    print(f"{'=' * 70}")
    os.makedirs(run_dir, exist_ok=True)
    set_seed(opts.seed)

    device = get_device()
    dtype = resolve_dtype(opts, training=True)
    bnb_config = build_bnb_config(opts, dtype) if method == "qlora" else None

    tokenizer = load_tokenizer(spec, model_type, model_name, opts.cache_dir)
    model = load_model(spec, model_type, model_name, tokenizer, dtype, bnb_config,
                       device_map="auto" if bnb_config is not None else None, cache_dir=opts.cache_dir)
    model = apply_method(model, spec, method, opts)
    if bnb_config is None and model_type != "mamba2":
        model.to(device)
    trainable_params, total_params = count_parameters(model)

    # ---- Data ----
    train_data = load_lamp_split(os.path.join(data_dir, "train"))
    val_data = _load_validation(data_dir, opts)
    epochs = opts.num_epochs
    if opts.debug:
        train_data, val_data, epochs = train_data[:8], val_data[:4], 1
        print("   *** DEBUG MODE: 8 train samples, 4 validation samples, 1 epoch ***")
    print(f"   Train samples: {len(train_data)}")

    device_str = str(device)
    if retriever:
        retriever.to_device(device_str)

    # ---- Optional zero-shot evaluation (same prompts, before any update) ----
    zero_shot_metrics = None
    if opts.eval_before_training:
        print("\nZero-shot evaluation before training...")
        zs_predictions = generate_predictions(model, tokenizer, val_data, task, model_type, retriever, opts)
        zero_shot_metrics = compute_task_metrics(zs_predictions, task)
        _print_metrics("Zero-shot metrics:", zero_shot_metrics)

    train_hf = build_hf_dataset(train_data, task, tokenizer, retriever, opts, "Building train prompts")
    val_hf = None if opts.no_epoch_eval else build_hf_dataset(val_data, task, tokenizer, retriever, opts,
                                                               "Building validation prompts")
    if retriever:
        retriever.to_cpu()  # free GPU memory for training

    preprocess = make_preprocessor(spec, model_type, tokenizer, opts.max_length)
    train_hf = train_hf.map(preprocess, batched=True, remove_columns=train_hf.column_names)
    if val_hf is not None:
        val_hf = val_hf.map(preprocess, batched=True, remove_columns=val_hf.column_names)

    # ---- Trainer ----
    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=run_dir,
        do_train=True,
        do_eval=val_hf is not None,
        eval_strategy="epoch" if val_hf is not None else "no",
        per_device_train_batch_size=opts.batch_size,
        per_device_eval_batch_size=opts.batch_size,
        gradient_accumulation_steps=opts.gradient_accumulation_steps,
        learning_rate=opts.learning_rate,
        weight_decay=opts.weight_decay,
        num_train_epochs=epochs,
        lr_scheduler_type="linear",
        warmup_ratio=opts.warmup_ratio,
        save_strategy="no",  # final model is saved manually
        logging_steps=opts.logging_steps,
        bf16=bool(opts.bf16 and use_cuda),
        fp16=bool(opts.fp16 and use_cuda),
        seed=opts.seed,
        report_to="none",
        gradient_checkpointing=False,
    )
    nvml_callback = NVMLTrainingCallback(
        output_dir=run_dir,
        gpu_index=opts.gpu_index,
        sample_interval_ms=opts.nvml_sample_interval,
        sample_every_n_steps=opts.nvml_sample_every_n_steps,
        use_background_sampling=not opts.nvml_no_background_sampling,
    )
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True,
                                             max_length=opts.max_length, label_pad_token_id=-100),
        train_dataset=train_hf,
        eval_dataset=val_hf,
        processing_class=tokenizer,
        callbacks=[nvml_callback],
    )
    if method == "loraplus":
        trainer = LoraPlusTrainer(**trainer_kwargs, loraplus_ratio=opts.loraplus_ratio)
    else:
        trainer = Trainer(**trainer_kwargs)

    print("\nStarting training...")
    start = time.time()
    train_result = trainer.train()
    training_time_s = time.time() - start
    final_train_loss = getattr(train_result, "training_loss", None)
    eval_losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
    print(f"   Training completed in {training_time_s / 60:.1f} minutes")
    nvml = nvml_callback.summary

    # ---- Post-training evaluation = tracked inference pass ----
    _free_training_state(trainer, model)
    if retriever:
        retriever.to_device(device_str)
    print("\nPost-training evaluation (inference pass, batch size 1)...")
    tracker = GPUInferenceTracker(gpu_index=opts.gpu_index)
    predictions = generate_predictions(model, tokenizer, val_data, task, model_type, retriever, opts, tracker)
    inference_stats = tracker.summary()
    metrics = compute_task_metrics(predictions, task)
    performance = netscore_performance(task, metrics)
    zs_performance = netscore_performance(task, zero_shot_metrics) if zero_shot_metrics else None

    # ---- FLOPs and NetScore ----
    flops = estimate_flops(spec, method, opts.max_length, opts.max_new_tokens, opts.num_beams,
                           trainable_params, total_params)
    netscore_inputs = to_netscore_inputs(
        performance=performance,
        params=trainable_params,
        flops=flops["training_flops_per_sequence"],
        vram_mb=nvml.get("max_vram_used_mb"),
        time_s=training_time_s,
        power_w=nvml.get("avg_power_watts"),
    )
    netscore = compute_netscore_variants(netscore_inputs)
    netscore_inputs_inference = to_netscore_inputs(
        performance=performance,
        params=trainable_params,
        flops=flops["inference_flops_per_sequence"],
        vram_mb=inference_stats.get("peak_vram_mb"),
        time_s=inference_stats.get("total_time_s"),
        power_w=inference_stats.get("avg_power_w"),
    )
    netscore_inference = compute_netscore_variants(netscore_inputs_inference)

    # ---- Save ----
    _save_json(os.path.join(run_dir, "predictions.json"), predictions)
    if inference_stats:
        _save_json(os.path.join(run_dir, "inference_stats.json"), inference_stats)
    if not opts.no_save_model:
        final_model_path = os.path.join(run_dir, "final_model")
        # Mamba2 has tied embedding/lm_head weights that safetensors rejects
        model.save_pretrained(final_model_path, safe_serialization=model_type != "mamba2")
        tokenizer.save_pretrained(final_model_path)
        print(f"   Saved {'adapter' if method in LORA_METHODS else 'model'}: {final_model_path}")

    results = {
        "mode": "train",
        "task": task,
        "task_description": task_cfg.description,
        "method": method,
        "model_type": model_type,
        "model_name": model_name,
        "hyperparameters": {
            "batch_size": opts.batch_size,
            "gradient_accumulation_steps": opts.gradient_accumulation_steps,
            "learning_rate": opts.learning_rate,
            "epochs": epochs,
            "warmup_ratio": opts.warmup_ratio,
            "weight_decay": opts.weight_decay,
            "max_length": opts.max_length,
            "max_new_tokens": opts.max_new_tokens,
            "num_beams": opts.num_beams if spec.kind == "seq2seq" else 1,
            "lora_r": opts.lora_r if method in LORA_METHODS else None,
            "lora_alpha": opts.lora_alpha if method in LORA_METHODS else None,
            "lora_dropout": opts.lora_dropout if method in LORA_METHODS else None,
            "loraplus_ratio": opts.loraplus_ratio if method == "loraplus" else None,
            "qlora_bits": opts.qlora_bits if method == "qlora" else None,
            "num_retrieved": opts.num_retrieved,
            "max_profile_size": opts.max_profile_size,
            "retrieval": retriever is not None,
            "precision": str(dtype).replace("torch.", ""),
            "seed": opts.seed,
        },
        # Task metrics
        "metrics": metrics,
        "primary_metric": task_cfg.primary_metric,
        "primary_metric_value": metrics.get(task_cfg.primary_metric),
        "netscore_performance": performance,
        "zero_shot_metrics": zero_shot_metrics,
        "zero_shot_netscore_performance": zs_performance,
        "performance_improvement": (performance - zs_performance
                                    if performance is not None and zs_performance is not None else None),
        "eval_losses_per_epoch": eval_losses,
        # Training cost
        "training_time_seconds": training_time_s,
        "training_time_minutes": training_time_s / 60.0,
        "final_training_loss": float(final_train_loss) if final_train_loss is not None else None,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100.0 * trainable_params / max(total_params, 1),
        "avg_gpu_power_watts": nvml.get("avg_power_watts"),
        "max_gpu_power_watts": nvml.get("max_power_watts"),
        "avg_gpu_vram_used_mb": nvml.get("avg_vram_used_mb"),
        "max_gpu_vram_used_mb": nvml.get("max_vram_used_mb"),
        "peak_torch_allocated_mb": nvml.get("peak_torch_allocated_mb"),
        "num_power_samples": nvml.get("num_power_samples", 0),
        "estimated_energy_Wh": nvml.get("estimated_energy_wh"),
        "energy_measurement_source": nvml.get("energy_source"),
        # FLOPs and NetScore
        "flops": flops,
        "NetScore": netscore,
        "netscore_inputs": netscore_inputs,
        "inference_stats": strip_per_sample(inference_stats),
        "NetScore_inference": netscore_inference,
        "netscore_inputs_inference": netscore_inputs_inference,
    }
    _save_json(os.path.join(run_dir, f"benchmark_results_{method}.json"), results)

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print(f"{METHOD_NAMES[method]} SUMMARY - {model_type} / {task}")
    print(f"{'=' * 70}")
    _print_metrics("Validation metrics:", metrics)
    print(f"   * NetScore performance a: {performance if performance is None else round(performance * 100, 2)} %")
    print(f"\nTraining cost:")
    print(f"   * Training time: {training_time_s / 60:.1f} min | trainable params: {trainable_params:,} "
          f"({results['trainable_percentage']:.2f}%)")
    if nvml.get("avg_power_watts") is not None:
        print(f"   * Avg power: {nvml['avg_power_watts']:.2f} W | max VRAM: {nvml['max_vram_used_mb']:.0f} MiB | "
              f"energy: {nvml['estimated_energy_wh']:.4f} Wh ({nvml['energy_source']})")
    else:
        print("   * No GPU power/VRAM readings (NVML unavailable): NS-E / NS-M / NS# will be N/A")
    print("\nNetScore (training cost):")
    print(format_netscore_results(netscore))
    _print_inference_stats(inference_stats)
    print("\nNetScore (inference cost):")
    print(format_netscore_results(netscore_inference))

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_evaluation(opts, model_type: str, task: str, method: Optional[str], data_dir: str,
                   model_path: Optional[str], save_dir: str,
                   retriever: Optional[ContrieverRetriever]) -> Dict[str, Any]:
    """Evaluate a fine-tuned checkpoint (method set) or the base model zero-shot (method None)."""
    spec = MODELS[model_type]
    task_cfg = TASKS[task]
    model_name = opts.model_name or spec.hf_name
    zero_shot = method is None
    _check_supported(model_type, method)

    label = "Zero-shot" if zero_shot else f"Fine-tuned ({METHOD_NAMES[method]})"
    print(f"\n{'=' * 70}")
    print(f"{label} evaluation | model={model_name} | task={task} ({task_cfg.description})")
    print(f"Data: {data_dir}\nCheckpoint: {model_path or '-'}\nResults: {save_dir}")
    print(f"{'=' * 70}")
    os.makedirs(save_dir, exist_ok=True)

    dtype = resolve_dtype(opts, training=False)
    bnb_config = build_bnb_config(opts, dtype) if method == "qlora" else None
    loads_full_model = method in ("full_ft", "bitfit")

    tokenizer = load_tokenizer(spec, model_type, model_name, opts.cache_dir)
    source = model_path if loads_full_model else model_name
    model = load_model(spec, model_type, source, tokenizer, dtype, bnb_config,
                       device_map="auto" if torch.cuda.is_available() else None, cache_dir=opts.cache_dir)
    if method in LORA_METHODS:
        print(f"   Loading adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    val_data = _load_validation(data_dir, opts)
    if retriever:
        retriever.to_device(str(get_device()))

    tracker = GPUInferenceTracker(gpu_index=opts.gpu_index)
    predictions = generate_predictions(model, tokenizer, val_data, task, model_type, retriever, opts, tracker)
    inference_stats = tracker.summary()
    metrics = compute_task_metrics(predictions, task)
    performance = netscore_performance(task, metrics)

    _, total_params = count_parameters(model)
    adapted_params = adapted_parameter_count(model, method)
    flops = estimate_flops(spec, method, opts.max_length, opts.max_new_tokens, opts.num_beams,
                           adapted_params, total_params)
    netscore_inputs_inference = to_netscore_inputs(
        performance=performance,
        params=adapted_params,
        flops=flops["inference_flops_per_sequence"],
        vram_mb=inference_stats.get("peak_vram_mb"),
        time_s=inference_stats.get("total_time_s"),
        power_w=inference_stats.get("avg_power_w"),
    )
    netscore_inference = compute_netscore_variants(netscore_inputs_inference)

    prefix = "zero_shot_" if zero_shot else ""
    results = {
        "mode": "zero_shot" if zero_shot else "evaluate",
        "task": task,
        "task_description": task_cfg.description,
        "method": method,
        "model_type": model_type,
        "model_name": model_name,
        "model_path": model_path,
        "settings": {
            "max_length": opts.max_length,
            "max_new_tokens": opts.max_new_tokens,
            "num_beams": opts.num_beams if spec.kind == "seq2seq" else 1,
            "num_retrieved": opts.num_retrieved,
            "max_profile_size": opts.max_profile_size,
            "retrieval": retriever is not None,
            "precision": str(dtype).replace("torch.", ""),
        },
        "metrics": metrics,
        "primary_metric": task_cfg.primary_metric,
        "primary_metric_value": metrics.get(task_cfg.primary_metric),
        "netscore_performance": performance,
        "total_parameters": total_params,
        "adapted_parameters": adapted_params,
        "flops": flops,
        "inference_stats": strip_per_sample(inference_stats),
        "NetScore_inference": netscore_inference,
        "netscore_inputs_inference": netscore_inputs_inference,
    }
    _save_json(os.path.join(save_dir, f"{prefix}eval_results.json"), results)
    _save_json(os.path.join(save_dir, f"{prefix}predictions.json"), predictions)
    if inference_stats:
        _save_json(os.path.join(save_dir, f"{prefix}inference_stats.json"), inference_stats)

    print(f"\n{'=' * 70}")
    print(f"{label.upper()} RESULTS - {model_type} / {task}")
    print(f"{'=' * 70}")
    _print_metrics("Validation metrics:", metrics)
    _print_inference_stats(inference_stats)
    print("\nNetScore (inference cost):")
    print(format_netscore_results(netscore_inference))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# =============================================================================
# CLI
# =============================================================================

def _normalize_task(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "")  # "LaMP-1" -> "lamp1"


def _parse_list(parser, value: str, choices, arg: str, normalize=lambda s: s.strip(), aliases=None) -> List[str]:
    if value.strip().lower() == "all":
        return list(choices)
    items = []
    for raw in value.split(","):
        if not raw.strip():
            continue
        item = normalize(raw)
        item = (aliases or {}).get(item, item)
        if item not in choices:
            parser.error(f"invalid {arg} '{raw.strip()}'. Choices: {', '.join(choices)}, all")
        if item not in items:
            items.append(item)
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LaMP Benchmark Framework: fine-tuning, evaluation, FLOPs and NetScore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'lamp_benchmark.py train -h' or 'lamp_benchmark.py evaluate -h'; see README.md for examples.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Fine-tune, evaluate, and score (NetScore training + inference cost)")
    evaluate = sub.add_parser("evaluate", help="Evaluate a checkpoint or the base model (--zero_shot)")

    for p in (train, evaluate):
        g = p.add_argument_group("model / task / data")
        g.add_argument("--model_type", required=True,
                       help=f"{', '.join(MODELS)} (alias: qwen), comma list, or all")
        g.add_argument("--model_name", "--base_model", dest="model_name", default=None,
                       help="Override the HF model id (single model_type only)")
        g.add_argument("--task", required=True,
                       help=f"{', '.join(TASKS)} (LaMP-1 style accepted), comma list, or all")
        g.add_argument("--method", default="lora", help=f"{', '.join(METHODS)}, comma list, or all (default: lora)")
        g.add_argument("--data_dir", required=True,
                       help="Dataset folder (with train/ and validation/) for one task, "
                            "or the parent of dataset1..dataset4 for several tasks")
        g.add_argument("--output_dir", required=(p is train), default=None,
                       help="Run output directory (sub-folders are added per swept dimension)")
        g.add_argument("--cache_dir", default="./cache", help="HF cache directory (default: ./cache)")

        g = p.add_argument_group("retrieval / prompting / generation")
        g.add_argument("--num_retrieved", type=int, default=4, help="Profile items to retrieve (default: 4)")
        g.add_argument("--max_profile_size", type=int, default=100,
                       help="Max profile items considered for retrieval (default: 100)")
        g.add_argument("--use_retrieval", action="store_true", default=True, help=argparse.SUPPRESS)
        g.add_argument("--no_retrieval", action="store_false", dest="use_retrieval",
                       help="Disable Contriever (use the first profile items)")
        g.add_argument("--max_length", type=int, default=512, help="Max sequence length / FLOPs S (default: 512)")
        g.add_argument("--max_new_tokens", type=int, default=32, help="Generated tokens / FLOPs T (default: 32)")
        g.add_argument("--num_beams", "--generation_num_beams", dest="num_beams", type=int, default=4,
                       help="Beam search width for flan-t5 (causal models use greedy) (default: 4)")
        g.add_argument("--max_samples", type=int, default=None, help="Limit validation samples")

        g = p.add_argument_group("hardware / precision")
        g.add_argument("--bf16", action="store_true", help="Load weights and train in bfloat16")
        g.add_argument("--fp16", action="store_true", help="Load weights and train in float16")
        g.add_argument("--qlora_bits", type=int, default=4, choices=[4, 8], help="QLoRA quantization bits")
        g.add_argument("--gpu_index", type=int, default=0, help="GPU index for NVML monitoring (default: 0)")
        g.add_argument("--seed", type=int, default=42)
        g.add_argument("--dry_run", action="store_true", help="Print planned runs and exit")

    g = train.add_argument_group("training")
    g.add_argument("--batch_size", type=int, default=4, help="Per-device batch size (default: 4)")
    g.add_argument("--gradient_accumulation_steps", type=int, default=4, help="(default: 4)")
    g.add_argument("--learning_rate", type=float, default=2e-4, help="(default: 2e-4)")
    g.add_argument("--num_epochs", type=int, default=10, help="(default: 10)")
    g.add_argument("--warmup_ratio", type=float, default=0.05, help="(default: 0.05)")
    g.add_argument("--weight_decay", type=float, default=0.01, help="(default: 0.01)")
    g.add_argument("--logging_steps", type=int, default=50, help="(default: 50)")
    g.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    g.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    g.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")
    g.add_argument("--loraplus_ratio", type=int, default=16, help="LoRA+ lr_B / lr_A (default: 16)")
    g.add_argument("--eval_before_training", action="store_true",
                   help="Also run a zero-shot generation pass before training (reports improvement)")
    g.add_argument("--no_epoch_eval", action="store_true", help="Skip the per-epoch validation loss")
    g.add_argument("--no_save_model", action="store_true", help="Do not save final_model/")
    g.add_argument("--debug", action="store_true", help="8 train / 4 validation samples, 1 epoch")
    g.add_argument("--nvml_sample_interval", type=int, default=100, help="Background sampling interval, ms")
    g.add_argument("--nvml_sample_every_n_steps", type=int, default=10, help="Also sample every N steps")
    g.add_argument("--nvml_no_background_sampling", action="store_true", help="Only sample at training steps")

    g = evaluate.add_argument_group("evaluation")
    g.add_argument("--model_path", "--lora_path", dest="model_path", default=None,
                   help="Adapter (lora/loraplus/qlora) or full checkpoint (full_ft/bitfit) directory")
    g.add_argument("--zero_shot", action="store_true", help="Evaluate the base model without fine-tuning")
    return parser


def _plan_runs(parser, opts) -> List[Dict[str, Any]]:
    model_types = _parse_list(parser, opts.model_type, list(MODELS), "model_type", aliases=MODEL_ALIASES)
    tasks = _parse_list(parser, opts.task, list(TASKS), "task", normalize=_normalize_task)
    zero_shot = opts.command == "evaluate" and opts.zero_shot
    methods = [None] if zero_shot else _parse_list(parser, opts.method, METHODS, "method")
    if opts.model_name and len(model_types) > 1:
        parser.error("--model_name can only be used with a single --model_type")
    if opts.bf16 and opts.fp16:
        parser.error("use only one of --bf16 / --fp16")

    n_runs = len(model_types) * len(tasks) * len(methods)
    if opts.command == "evaluate" and not zero_shot and opts.model_path and n_runs > 1:
        parser.error("--model_path evaluates one checkpoint; for sweeps pass --output_dir of the training "
                     "sweep instead (each run's <run_dir>/final_model is used)")
    if opts.command == "evaluate" and not zero_shot and not opts.model_path and not opts.output_dir:
        parser.error("evaluate needs --model_path, or --output_dir containing final_model/ (or --zero_shot)")

    runs = []
    for model_type in model_types:
        for task in tasks:
            for method in methods:
                parts = [opts.output_dir or "."]
                if len(model_types) > 1:
                    parts.append(model_type)
                if len(tasks) > 1:
                    parts.append(task)
                if len(methods) > 1:
                    parts.append(method)
                run_dir = os.path.join(*parts)

                run = {"model_type": model_type, "task": task, "method": method, "run_dir": run_dir,
                       "data_dir": resolve_task_data_dir(opts.data_dir, task, len(tasks) > 1),
                       "model_path": None, "save_dir": run_dir}
                if opts.command == "evaluate":
                    if zero_shot:
                        if not opts.output_dir:
                            run["save_dir"] = f"./zero_shot_{model_type}_{task}"
                    elif opts.model_path:
                        run["model_path"] = opts.model_path
                        path = opts.model_path.rstrip("/")
                        run["save_dir"] = opts.output_dir or (
                            os.path.dirname(path) or "." if os.path.basename(path) == "final_model" else path)
                    else:
                        final_model = os.path.join(run_dir, "final_model")
                        run["model_path"] = final_model if os.path.isdir(final_model) else run_dir
                runs.append(run)
    return runs


def _print_summary_table(all_results: List[Dict[str, Any]], key: str, title: str):
    rows = [r for r in all_results if r.get("status") == "success" and r.get(key)]
    if not rows:
        return
    ns_header = " ".join(f"{name:>8}" for name in NETSCORE.variants)
    header = f"{'Model':<10} {'Task':<6} {'Method':<10} {'Metric':<9} {'Value':>8} {'a(%)':>7} {ns_header}"
    print(f"\n{title}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        value = r.get("primary_metric_value")
        perf = r.get("netscore_performance")
        ns = " ".join(f"{r[key][n]:>8.2f}" if r[key].get(n) is not None else f"{'N/A':>8}"
                      for n in NETSCORE.variants)
        print(f"{r['model_type']:<10} {r['task']:<6} {str(r['method'] or 'zero_shot'):<10} "
              f"{r['primary_metric']:<9} {value if value is None else format(value, '8.4f'):>8} "
              f"{'N/A' if perf is None else format(perf * 100, '.2f'):>7} {ns}")
    print("-" * len(header))


def main():
    parser = build_parser()
    opts = parser.parse_args()
    runs = _plan_runs(parser, opts)

    print("=" * 70)
    print(f"LaMP Benchmark Framework - {opts.command}")
    print("=" * 70)
    print(f"Planned runs: {len(runs)}")
    for i, run in enumerate(runs, 1):
        extra = f" | checkpoint: {run['model_path']}" if run["model_path"] else ""
        print(f"   {i}. {run['model_type']} + {run['task']} + {run['method'] or 'zero_shot'} "
              f"| data: {run['data_dir']} | out: {run['save_dir']}{extra}")
    print(f"Retrieval: {'Contriever' if opts.use_retrieval else 'disabled'} | NVML: {NVML_OK} | "
          f"device: {get_device()}")
    if opts.dry_run:
        print("\nDry run - nothing executed.")
        return

    os.makedirs(opts.cache_dir, exist_ok=True)
    retriever = None
    if opts.use_retrieval:
        print("\nLoading Contriever retriever...")
        retriever = ContrieverRetriever(device=str(get_device()))

    all_results: List[Dict[str, Any]] = []
    for i, run in enumerate(runs, 1):
        print(f"\n>>> Run {i}/{len(runs)}: {run['model_type']} + {run['task']} + {run['method'] or 'zero_shot'}")
        try:
            if not os.path.isdir(os.path.join(run["data_dir"], "validation")):
                raise FileNotFoundError(f"validation/ not found in data directory: {run['data_dir']}")
            if opts.command == "train":
                result = run_training(opts, run["model_type"], run["task"], run["method"],
                                      run["data_dir"], run["run_dir"], retriever)
            else:
                if run["method"] is not None and not os.path.exists(run["model_path"]):
                    raise FileNotFoundError(f"checkpoint not found: {run['model_path']}")
                result = run_evaluation(opts, run["model_type"], run["task"], run["method"],
                                        run["data_dir"], run["model_path"], run["save_dir"], retriever)
            result["status"] = "success"
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nRun failed: {e}")
            result = {"model_type": run["model_type"], "task": run["task"], "method": run["method"],
                      "status": "failed", "error": str(e)}
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_results.append(result)

    ok = sum(1 for r in all_results if r["status"] == "success")
    print(f"\n{'=' * 70}\nBENCHMARK COMPLETE: {ok}/{len(runs)} successful\n{'=' * 70}")
    for r in all_results:
        if r["status"] == "failed":
            print(f"   FAILED {r['model_type']} + {r['task']} + {r['method']}: {r['error']}")
    if opts.command == "train":
        _print_summary_table(all_results, "NetScore", "NetScore (training cost):")
    _print_summary_table(all_results, "NetScore_inference", "NetScore (inference cost):")

    if len(runs) > 1:
        os.makedirs(opts.output_dir or ".", exist_ok=True)
        _save_json(os.path.join(opts.output_dir or ".", f"all_results_{opts.command}.json"), all_results)


if __name__ == "__main__":
    main()
