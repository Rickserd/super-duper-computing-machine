#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics utilities including NetScore calculation.

NetScore = S · log10( a^α / ((p·m)^β · v^γ · t^δ · w^λ) )

Where:
- a: task performance (accuracy, MCC for CoLA, combined Pearson/Spearman for STS-B)
- p: number of trainable parameters
- m: FLOPs per sequence (see utils/flops.py)
- v: peak VRAM
- t: fine-tuning (or inference) time
- w: average power draw
- S, α: scale and performance exponent (S = 20, α = 2)
- β, γ, δ, λ: efficiency exponents that switch cost terms on or off per variant
  (NS, NS-E, NS-M, NS#; see NetScoreConfig in config.py)
"""

import math
from typing import Dict, Optional

import numpy as np

from config import NETSCORE, NetScoreConfig


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

    # Evaluated in log space: S · (α·log a − β·log p − β·log m − γ·log v − δ·log t − λ·log w)
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
    """
    Convert raw measurements into NetScore units.

    Args:
        performance: Primary task metric in [0, 1] (or [-1, 1] for correlations)
        params: Trainable parameter count
        flops: FLOPs per sequence
        vram_mb: Peak VRAM in MiB
        time_s: Fine-tuning or inference time in seconds
        power_w: Average power draw in Watts
        cfg: NetScore configuration holding the unit conversions

    Returns:
        Dictionary with keys a, p, m, v, t, w
    """
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
    """
    Compute every configured NetScore variant (NS, NS-E, NS-M, NS#).

    Args:
        inputs: Output of ``to_netscore_inputs``
        cfg: NetScore configuration (scale, alpha, variant exponents)

    Returns:
        Dictionary mapping variant name to its value (None if not computable)
    """
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


def compute_metrics_for_dataset(
    eval_pred,
    metric_name: str,
):
    """
    Compute metrics for a given dataset.

    Args:
        eval_pred: Tuple of (logits, labels)
        metric_name: Name of the metric to compute
            ('accuracy', 'matthews_correlation', or 'pearson_spearman')

    Returns:
        Dictionary with computed metrics
    """
    import evaluate

    logits, labels = eval_pred

    if metric_name == "pearson_spearman":
        preds = logits.squeeze(-1)
        pearson_metric = evaluate.load("pearsonr")
        spearman_metric = evaluate.load("spearmanr")
        pearson = pearson_metric.compute(predictions=preds, references=labels)["pearsonr"]
        spearman = spearman_metric.compute(predictions=preds, references=labels)["spearmanr"]
        return {
            "pearson": pearson,
            "spearman": spearman,
            "combined_score": (pearson + spearman) / 2.0,
        }

    preds = np.argmax(logits, axis=1)

    if metric_name == "accuracy":
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=preds, references=labels)
    elif metric_name == "matthews_correlation":
        metric = evaluate.load("matthews_correlation")
        return metric.compute(predictions=preds, references=labels)
    else:
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=preds, references=labels)


def format_netscore_results(netscore: Dict[str, Optional[float]]) -> str:
    """Format NetScore variants for display."""
    lines = []
    for key, value in netscore.items():
        if value is not None:
            lines.append(f"   • {key}: {value:.4f}")
        else:
            lines.append(f"   • {key}: N/A (missing or non-positive inputs)")
    return "\n".join(lines)
