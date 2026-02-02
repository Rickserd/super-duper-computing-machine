#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility to discover linear modules in models for LoRA targeting.
"""

import torch.nn as nn
from typing import List, Set


def find_all_linear_names(model: nn.Module, include_lm_head: bool = False) -> List[str]:
    """
    Find all linear layer names in a model that can be targeted by LoRA.
    
    Args:
        model: The model to inspect
        include_lm_head: Whether to include the LM head / output projection
        
    Returns:
        List of module names that are Linear layers
    """
    linear_names = set()
    
    # Module names to exclude (usually frozen or special purpose)
    exclude_keywords = {"embed", "norm", "score"}  # embedding, layer norms, classification head
    if not include_lm_head:
        exclude_keywords.add("lm_head")
        exclude_keywords.add("output")
    
    for name, module in model.named_modules():
        # Check if it's a linear layer
        if isinstance(module, nn.Linear):
            # Get the last part of the name (the actual layer name)
            name_parts = name.split(".")
            layer_name = name_parts[-1]
            
            # Skip excluded modules
            skip = False
            for keyword in exclude_keywords:
                if keyword in name.lower():
                    skip = True
                    break
            
            if not skip:
                linear_names.add(layer_name)
    
    return sorted(list(linear_names))


def find_target_modules_for_lora(model: nn.Module) -> List[str]:
    """
    Find appropriate target modules for LoRA based on common patterns.
    
    Prioritizes attention-related modules, then falls back to all linear layers.
    
    Args:
        model: The model to inspect
        
    Returns:
        List of module names suitable for LoRA
    """
    all_linear = find_all_linear_names(model)
    
    # Common attention projection names across different architectures
    attention_patterns = [
        # Standard transformer (LLaMA, Qwen, etc.)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Alternative naming
        "query", "key", "value", "dense",
        # GPT-style
        "c_attn", "c_proj",
        # BERT-style
        "query", "key", "value", "output.dense",
        # Mamba / SSM
        "in_proj", "out_proj", "x_proj", "dt_proj",
        # MLP
        "gate_proj", "up_proj", "down_proj",
        "fc1", "fc2", "mlp.fc1", "mlp.fc2",
    ]
    
    # Find which attention patterns exist in the model
    found_modules = []
    for pattern in attention_patterns:
        if pattern in all_linear:
            found_modules.append(pattern)
    
    if found_modules:
        return found_modules
    
    # Fallback: return all linear layers (except excluded ones)
    return all_linear


def print_model_modules(model: nn.Module, filter_type: str = None):
    """
    Print all modules in a model for inspection.
    
    Args:
        model: The model to inspect
        filter_type: Optional filter - "linear", "attention", etc.
    """
    print("=" * 80)
    print("Model Modules:")
    print("=" * 80)
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        if filter_type:
            if filter_type.lower() == "linear" and not isinstance(module, nn.Linear):
                continue
        
        if isinstance(module, nn.Linear):
            print(f"[Linear] {name}: {module.in_features} -> {module.out_features}")
        elif "attention" in module_type.lower() or "attn" in name.lower():
            print(f"[Attention] {name}: {module_type}")
        elif not filter_type:
            print(f"  {name}: {module_type}")
    
    print("=" * 80)
    print(f"\nLinear layer names suitable for LoRA:")
    print(find_all_linear_names(model))
    print(f"\nRecommended LoRA target modules:")
    print(find_target_modules_for_lora(model))




