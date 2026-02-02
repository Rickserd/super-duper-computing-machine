#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration for models, datasets, and training parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ============================================================================
# Model Configurations
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    hf_name: str
    target_modules: List[str]  # For LoRA
    description: str = ""
    # If True, model doesn't support AutoModelForSequenceClassification natively
    # and needs a custom classification head wrapper on top of CausalLM
    requires_custom_head: bool = False
    # Padding side: "right" (default for most models) or "left" (for some models like Hymba)
    padding_side: str = "right"
    # If True, pad all sequences to max_length during tokenization (required for some models like Hymba)
    pad_to_max_length: bool = False
    # If True, use Flash Attention 2 for faster attention computation (requires flash_attn package)
    use_flash_attn: bool = False


MODELS: Dict[str, ModelConfig] = {
    "tinyllama-1.1b": ModelConfig(
        name="tinyllama-1.1b",
        hf_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="TinyLlama 1.1B Chat model",
    ),
    "qwen3-1.7b": ModelConfig(
        name="qwen3-1.7b",
        hf_name="Qwen/Qwen3-1.7B",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Qwen3 1.7B model",
    ),
    "hymba-1.5b": ModelConfig(
        name="hymba-1.5b",
        hf_name="nvidia/Hymba-1.5B-Instruct",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="NVIDIA Hymba 1.5B Instruct model (hybrid Mamba-Attention)",
        requires_custom_head=True,  # Uses custom architecture, needs CausalLM + classification head
        padding_side="right",  # Use right-padding to avoid Hymba's shift_zeros_to_front issues
        pad_to_max_length=False,  # Use dynamic padding for efficiency (not fixed-length)
        use_flash_attn=True,  # Hymba benefits from Flash Attention 2 for faster attention
    ),
}


# ============================================================================
# Dataset Configurations
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    hf_name: str
    subset: Optional[str]
    text_column: str
    text_column_2: Optional[str]  # For sentence-pair tasks
    label_column: str
    num_labels: int
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    metric_name: str  # Primary metric to use
    metric_for_best_model: str  # Metric name as returned by evaluate
    greater_is_better: bool = True
    description: str = ""


DATASETS: Dict[str, DatasetConfig] = {
    "sst2": DatasetConfig(
        name="sst2",
        hf_name="glue",
        subset="sst2",
        text_column="sentence",
        text_column_2=None,
        label_column="label",
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        metric_name="accuracy",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        description="Stanford Sentiment Treebank v2 (binary sentiment)",
    ),
    "qnli": DatasetConfig(
        name="qnli",
        hf_name="glue",
        subset="qnli",
        text_column="question",
        text_column_2="sentence",
        label_column="label",
        num_labels=2,
        id2label={0: "entailment", 1: "not_entailment"},
        label2id={"entailment": 0, "not_entailment": 1},
        metric_name="accuracy",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        description="Question-answering NLI (derived from SQuAD)",
    ),
    "cola": DatasetConfig(
        name="cola",
        hf_name="glue",
        subset="cola",
        text_column="sentence",
        text_column_2=None,
        label_column="label",
        num_labels=2,
        id2label={0: "unacceptable", 1: "acceptable"},
        label2id={"unacceptable": 0, "acceptable": 1},
        metric_name="matthews_correlation",  # MCC for CoLA
        metric_for_best_model="eval_matthews_correlation",
        greater_is_better=True,
        description="Corpus of Linguistic Acceptability (MCC metric)",
    ),
}


# ============================================================================
# Training Methods
# ============================================================================

METHODS = ["bitfit", "full_ft", "lora", "loraplus", "qlora"]


# ============================================================================
# Default Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Default training configuration."""
    seed: int = 42  # Random seed for reproducibility
    batch_size: int = 32
    learning_rate: float = 1e-5
    epochs: int = 5
    max_length: int = 256
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 1
    gradient_accumulation_steps: int = 1
    gpu_index: int = 0
    
    # LoRA specific
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # LoRA+ specific
    loraplus_ratio: int = 16
    
    # NVML Power sampling configuration
    # Use continuous background sampling for accurate power measurement during training
    nvml_use_background_sampling: bool = True
    nvml_sample_interval_ms: int = 100  # 100ms = 10 samples/sec
    nvml_sample_every_n_steps: int = 10  # Also sample at every N training steps


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name]


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration by name."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    return DATASETS[dataset_name]

