#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seed utilities for reproducibility.
"""

import os
import random


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
        deterministic: If True, enable fully deterministic operations.
                      This may significantly slow down training but ensures
                      exact reproducibility. Default is False.
    """
    # Python's built-in random
    random.seed(seed)
    
    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            
            if deterministic:
                # Enable deterministic operations (slower but reproducible)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                # For PyTorch >= 1.8
                if hasattr(torch, "use_deterministic_algorithms"):
                    try:
                        torch.use_deterministic_algorithms(True)
                    except Exception:
                        pass
            else:
                # Allow cuDNN to find the best algorithm (faster but non-deterministic)
                torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
    
    # Transformers
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass
    
    print(f"🎲 Random seed set to: {seed}" + (" (deterministic mode)" if deterministic else ""))






