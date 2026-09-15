#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration for models, datasets, and training parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch


# ============================================================================
# Model Configurations
# ============================================================================

@dataclass
class ArchSpec:
    """Architecture shapes used by the analytical FLOPs model (utils/flops.py)."""
    family: str  # "transformer" | "mamba1" | "mamba2"
    num_layers: int
    hidden_size: int  # d_model for Mamba
    # Transformer (num_heads is also the SSM head count for Mamba-2)
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    # Mamba
    d_inner: int = 0
    d_state: int = 0
    d_conv: int = 4
    dt_rank: int = 0
    ngroups: int = 1


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    hf_name: str
    target_modules: List[str]  # For LoRA
    description: str = ""
    requires_custom_head: bool = False  # True for models without native SeqCls support
    requires_mamba_ssm: bool = False  # True for models loaded via mamba_ssm package (Mamba2)
    tokenizer_name: Optional[str] = None  # Override tokenizer (defaults to hf_name)
    padding_side: str = "right"  # Padding side ("left" required for Mamba2)
    modules_to_save: Optional[List[str]] = None  # Modules kept fully trainable in PEFT
    torch_dtype: Optional[torch.dtype] = None  # Override dtype for loading (e.g. bf16 for fp32 models)
    quantization_skip_modules: Optional[List[str]] = None  # Modules to keep in full precision during quantization
    quantized_model_path: Optional[str] = None  # Path to pre-quantized model (for QLoRA with Mamba etc.)
    arch: Optional[ArchSpec] = None  # Shapes for analytical FLOPs (None -> 2 * params * seq_len approximation)


MODELS: Dict[str, ModelConfig] = {
    "tinyllama-1.1b": ModelConfig(
        name="tinyllama-1.1b",
        hf_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="TinyLlama 1.1B Chat model",
        arch=ArchSpec(
            family="transformer", num_layers=22, hidden_size=2048,
            num_heads=32, num_kv_heads=4, head_dim=64, intermediate_size=5632,
        ),
    ),
    "qwen3-1.7b": ModelConfig(
        name="qwen3-1.7b",
        hf_name="Qwen/Qwen3-1.7B",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Qwen3 1.7B model",
        arch=ArchSpec(
            family="transformer", num_layers=28, hidden_size=2048,
            num_heads=16, num_kv_heads=8, head_dim=128, intermediate_size=6144,
        ),
    ),
    "mamba-1.4b": ModelConfig(
        name="mamba-1.4b",
        hf_name="state-spaces/mamba-1.4b-hf",
        target_modules=["x_proj", "in_proj"],
        description="Mamba-1 1.4B SSM model (fp32 weights, finetuned in bf16)",
        requires_custom_head=True,
        torch_dtype=torch.bfloat16,
        quantization_skip_modules=["dt_proj"],  # dt_proj.weight accessed directly in Mamba mixer
        quantized_model_path="./quantized_models/mamba-1.4b-gptq-4bit",  # GPTQ pre-quantized for QLoRA
        arch=ArchSpec(
            family="mamba1", num_layers=48, hidden_size=2048,
            d_inner=4096, d_state=16, d_conv=4, dt_rank=128,
        ),
    ),
    "mamba2-1.3b": ModelConfig(
        name="mamba2-1.3b",
        hf_name="state-spaces/mamba2-1.3b",
        target_modules=["in_proj", "x_proj"],  # NOT out_proj/conv1d — fused CUDA kernels bypass LoRA
        description="Mamba-2 1.3B SSM model (requires mamba_ssm package)",
        requires_mamba_ssm=True,
        tokenizer_name="EleutherAI/gpt-neox-20b",
        padding_side="left",  # Right-padding propagates noise in Mamba2
        modules_to_save=["classifier"],  # Keep classification head fully trainable in PEFT
        torch_dtype=torch.bfloat16,
        arch=ArchSpec(
            family="mamba2", num_layers=48, hidden_size=2048,
            num_heads=64, d_inner=4096, d_state=128, d_conv=4, ngroups=1,
        ),
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
    is_regression: bool = False  # True for regression tasks like STS-B
    description: str = ""
    # Multiple-choice tasks (e.g. HellaSwag) are reformulated as single-sequence
    # N-way classification: the context and all candidate endings are rendered into
    # one `text_column` string, and `label_column` is the gold choice index.
    is_multiple_choice: bool = False
    context_column: Optional[str] = None  # Raw column holding the context/premise
    choices_column: Optional[str] = None  # Raw column holding the list of candidate endings


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
    "stsb": DatasetConfig(
        name="stsb",
        hf_name="glue",
        subset="stsb",
        text_column="sentence1",
        text_column_2="sentence2",
        label_column="label",
        num_labels=1,
        id2label={0: "similarity"},
        label2id={"similarity": 0},
        metric_name="pearson_spearman",
        metric_for_best_model="eval_combined_score",
        greater_is_better=True,
        is_regression=True,
        description="Semantic Textual Similarity Benchmark (Pearson/Spearman correlation)",
    ),
    "hellaswag": DatasetConfig(
        name="hellaswag",
        hf_name="Rowan/hellaswag",
        subset=None,
        text_column="text",  # Synthesized from context + endings during preprocessing
        text_column_2=None,
        label_column="label",
        num_labels=4,
        id2label={0: "A", 1: "B", 2: "C", 3: "D"},
        label2id={"A": 0, "B": 1, "C": 2, "D": 3},
        metric_name="accuracy",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        is_multiple_choice=True,
        context_column="ctx",
        choices_column="endings",
        description="HellaSwag commonsense NLI (4-way multiple choice)",
    ),
}


# ============================================================================
# Training Methods
# ============================================================================

METHODS = ["bitfit", "full_ft", "lora", "loraplus", "qlora"]


# ============================================================================
# NetScore Configuration
# ============================================================================

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
    # Units: a = metric * performance_scale; every other raw value is divided by its unit
    performance_scale: float = 100.0  # a in %
    params_unit: float = 1e6          # p in millions
    flops_unit: float = 1e6           # m in millions of FLOPs
    vram_mb_unit: float = 1024.0      # v in GiB (raw values are MiB)
    time_s_unit: float = 1.0          # t in seconds
    power_w_unit: float = 1.0         # w in Watts


NETSCORE = NetScoreConfig()


# ============================================================================
# Default Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Default training configuration."""
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
    
    # Model saving
    save_model: bool = True  # Whether to save the final fine-tuned model


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

