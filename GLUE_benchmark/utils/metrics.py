#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics utilities including SAM (Sustainable AI Metric) calculation.

SAM(a) = acc^a × a / log(Wh)

Where:
- acc: accuracy (or other performance metric, e.g., MCC for CoLA)
- a: exponent parameter (commonly 1, 2, or 5)
- Wh: energy consumption in Watt-hours
"""

import math
from typing import Dict, Optional, List
import numpy as np


def calculate_sam(
    accuracy: float,
    energy_wh: float,
    alpha: int = 1,
    min_energy_wh: float = 1e-6,
) -> Optional[float]:
    """
    Calculate SAM (Sustainable AI Metric).
    
    SAM(a) = acc^a × a / log(Wh)
    
    Args:
        accuracy: Performance metric (accuracy, MCC, etc.) in [0, 1] range
        energy_wh: Energy consumption in Watt-hours
        alpha: Exponent parameter (commonly 1, 2, or 5)
        min_energy_wh: Minimum energy value to avoid log(0) issues
        
    Returns:
        SAM value or None if calculation is not possible
    """
    if energy_wh is None or energy_wh <= 0:
        return None
    
    # Ensure energy is at least min_energy_wh to avoid log issues
    energy_wh = max(energy_wh, min_energy_wh)
    
    # Avoid log(1) = 0 which would cause division by zero
    if energy_wh <= 1.0:
        # Use natural log shifted by 1 to ensure positive denominator
        # log(Wh + 1) for small values
        log_energy = math.log(energy_wh + 1)
    else:
        log_energy = math.log(energy_wh)
    
    if log_energy == 0:
        return None
    
    # SAM(a) = acc^a × a / log(Wh)
    sam = (accuracy ** alpha) * alpha / log_energy
    
    return sam


def compute_sam_metrics(
    accuracy: float,
    energy_wh: Optional[float],
    alphas: List[int] = [1, 2, 5],
) -> Dict[str, Optional[float]]:
    """
    Compute SAM metrics for multiple alpha values.
    
    Args:
        accuracy: Performance metric in [0, 1] range
        energy_wh: Energy consumption in Watt-hours
        alphas: List of alpha values to compute SAM for
        
    Returns:
        Dictionary with SAM values for each alpha
    """
    results = {}
    
    for alpha in alphas:
        key = f"SAM@{alpha}"
        if energy_wh is not None and energy_wh > 0:
            results[key] = calculate_sam(accuracy, energy_wh, alpha)
        else:
            results[key] = None
    
    return results


def compute_metrics_for_dataset(
    eval_pred,
    metric_name: str,
):
    """
    Compute metrics for a given dataset.
    
    Args:
        eval_pred: Tuple of (logits, labels)
        metric_name: Name of the metric to compute ('accuracy' or 'matthews_correlation')
        
    Returns:
        Dictionary with computed metrics
    """
    import evaluate
    
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    if metric_name == "accuracy":
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=preds, references=labels)
    elif metric_name == "matthews_correlation":
        metric = evaluate.load("matthews_correlation")
        return metric.compute(predictions=preds, references=labels)
    else:
        # Default to accuracy
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=preds, references=labels)


def format_sam_results(sam_metrics: Dict[str, Optional[float]]) -> str:
    """Format SAM metrics for display."""
    lines = []
    for key, value in sam_metrics.items():
        if value is not None:
            lines.append(f"   • {key}: {value:.6f}")
        else:
            lines.append(f"   • {key}: N/A (no energy data)")
    return "\n".join(lines)

