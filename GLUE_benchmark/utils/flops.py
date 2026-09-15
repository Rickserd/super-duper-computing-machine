#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analytical FLOPs model used for the m term of NetScore.

Cost model for sequence classification (the tiny classification head is
ignored). For one sequence of length S:

  F = forward FLOPs of the base model
  A = forward FLOPs of the LoRA adapters ~ 2 * S * trainable_params

  method                 training FLOPs / sequence
  ---------------------  -------------------------
  full_ft                3F
  bitfit                 2F   (bias grads ~0)
  lora / loraplus        2F + 3A
  qlora                  2F + 3A  (4-bit saves memory, not matmuls)

Inference is a single forward pass: F (adapters assumed merged).

A matmul producing an (m x n) output from a k-dim contraction costs 2*m*k*n
FLOPs. Models without an ArchSpec fall back to F ~ 2 * total_params * S.
"""

from typing import Any, Dict, Optional

from config import ArchSpec, ModelConfig

LORA_METHODS = ("lora", "loraplus", "qlora")


def _transformer_forward_flops(a: ArchSpec, S: int) -> float:
    d, hd, d_ff = a.hidden_size, a.head_dim, a.intermediate_size
    q_dim = a.num_heads * hd
    kv_dim = a.num_kv_heads * hd

    attn_proj = 2 * S * d * q_dim          # q_proj
    attn_proj += 2 * 2 * S * d * kv_dim    # k_proj + v_proj (GQA)
    attn_proj += 2 * S * q_dim * d         # o_proj
    attn_core = 2 * (2 * a.num_heads * S * S * hd)  # QK^T and softmax(.)V
    mlp = 3 * (2 * S * d * d_ff)           # gate, up, down (SwiGLU)

    return float(a.num_layers * (attn_proj + attn_core + mlp))


def _mamba1_forward_flops(a: ArchSpec, S: int) -> float:
    d, di, ds = a.hidden_size, a.d_inner, a.d_state
    in_proj = 2 * S * d * (2 * di)             # x and z
    conv = 2 * S * di * a.d_conv               # depthwise causal conv
    x_proj = 2 * S * di * (a.dt_rank + 2 * ds) # -> dt, B, C
    dt_proj = 2 * S * a.dt_rank * di
    scan = 9 * S * di * ds                     # selective scan recurrence (approx.)
    out_proj = 2 * S * di * d
    return float(a.num_layers * (in_proj + conv + x_proj + dt_proj + scan + out_proj))


def _mamba2_forward_flops(a: ArchSpec, S: int) -> float:
    d, di, ds, ng = a.hidden_size, a.d_inner, a.d_state, a.ngroups
    in_proj = 2 * S * d * (2 * di + 2 * ng * ds + a.num_heads)  # z, x, B, C, dt
    conv = 2 * S * (di + 2 * ng * ds) * a.d_conv
    scan = 9 * S * di * ds                     # SSD scan (approx.)
    out_proj = 2 * S * di * d
    return float(a.num_layers * (in_proj + conv + scan + out_proj))


def forward_flops(arch: ArchSpec, seq_len: int) -> float:
    """Forward FLOPs of the base model for one sequence of ``seq_len`` tokens."""
    if arch.family == "transformer":
        return _transformer_forward_flops(arch, seq_len)
    if arch.family == "mamba1":
        return _mamba1_forward_flops(arch, seq_len)
    if arch.family == "mamba2":
        return _mamba2_forward_flops(arch, seq_len)
    raise ValueError(f"Unknown architecture family: {arch.family}")


def training_flops(
    forward: Optional[float],
    method: str,
    seq_len: int,
    trainable_params: Optional[int],
    total_params: Optional[int] = None,
) -> Optional[float]:
    """Training (forward + backward) FLOPs for one sequence."""
    if forward is None:
        return None
    if method == "full_ft":
        return 3.0 * forward
    if method == "bitfit":
        return 2.0 * forward
    if trainable_params is None:
        return None
    # Methods outside the table: full backward if every weight is trained,
    # otherwise a frozen base with trainable adapters.
    if method not in LORA_METHODS and total_params and trainable_params >= total_params:
        return 3.0 * forward
    adapter = 2.0 * seq_len * trainable_params
    return 2.0 * forward + 3.0 * adapter


def estimate_flops(
    model_config: Optional[ModelConfig],
    method: str,
    seq_len: int,
    trainable_params: Optional[int],
    total_params: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate per-sequence forward and training FLOPs for a run.

    Args:
        model_config: Model configuration (its ``arch`` selects the analytical model)
        method: Fine-tuning method name
        seq_len: Sequence length (max_length)
        trainable_params: Trainable parameter count (sizes the LoRA adapter term)
        total_params: Total parameter count (used by the fallback approximation)

    Returns:
        Dict with ``seq_len``, ``forward_flops_per_sequence``,
        ``training_flops_per_sequence`` and ``source``
        ("analytical", "approx_2NS" or "unavailable").
    """
    arch = getattr(model_config, "arch", None)
    if arch is not None:
        forward, source = forward_flops(arch, seq_len), "analytical"
    elif total_params:
        forward, source = 2.0 * total_params * seq_len, "approx_2NS"
    else:
        forward, source = None, "unavailable"

    return {
        "seq_len": seq_len,
        "forward_flops_per_sequence": forward,
        "training_flops_per_sequence": training_flops(
            forward, method, seq_len, trainable_params, total_params
        ),
        "source": source,
    }
