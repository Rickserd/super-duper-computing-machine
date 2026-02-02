#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base trainer class for all fine-tuning methods.
"""

import os
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from config import ModelConfig, DatasetConfig, TrainingConfig
from utils.nvml_callback import CheckpointNVMLCallback, NVML_OK
from utils.metrics import compute_sam_metrics, format_sam_results


class BaseTrainer(ABC):
    """Base class for all training methods."""
    
    method_name: str = "base"
    
    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        output_dir: str,
        verbose: bool = True,
    ):
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Will be set during setup
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.tokenized_dataset = None
        self.trainer = None
        self.nvml_callback = None
        
        # Results
        self.results: Dict[str, Any] = {}
        self.training_time_seconds: float = 0.0
        
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def setup(self):
        """Setup all components for training."""
        self.log(f"\n{'='*70}")
        self.log(f"🚀 {self.method_name.upper()} Fine-tuning")
        self.log(f"   Model: {self.model_config.hf_name}")
        self.log(f"   Dataset: {self.dataset_config.name}")
        self.log(f"   NVML available: {NVML_OK}")
        self.log(f"{'='*70}")
        
        self._load_dataset()
        self._load_tokenizer()
        self._load_model()
        self._apply_method_specific_setup()
        self._tokenize_dataset()
        self._setup_trainer()
    
    def _load_dataset(self):
        """Load the dataset."""
        self.log(f"\n📁 Loading {self.dataset_config.name} dataset...")
        
        if self.dataset_config.subset:
            self.dataset = load_dataset(
                self.dataset_config.hf_name, 
                self.dataset_config.subset
            )
        else:
            self.dataset = load_dataset(self.dataset_config.hf_name)
        
        # Handle different split names
        train_split = "train"
        eval_split = "validation" if "validation" in self.dataset else "test"
        
        self.log(f"   Train samples: {len(self.dataset[train_split])}")
        self.log(f"   Eval samples: {len(self.dataset[eval_split])}")
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        self.log(f"\n🔤 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.hf_name, 
            use_fast=True,
            trust_remote_code=True,
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Use model-specific padding side (some models like Hymba require left-padding)
        self.tokenizer.padding_side = self.model_config.padding_side
        self.log(f"   ✓ Tokenizer loaded (padding_side={self.model_config.padding_side})")
    
    def _load_model(self):
        """Load the base model. Override in subclasses for quantized loading."""
        self.log(f"\n🧠 Loading base model...")
        
        if self.model_config.requires_custom_head:
            # Model doesn't support AutoModelForSequenceClassification natively
            # Use CausalLM with custom classification head
            self.log(f"   ℹ️  Model requires custom classification head (CausalLM wrapper)")
            if self.model_config.use_flash_attn:
                self.log(f"   ⚡ Using Flash Attention 2")
            from utils.custom_models import load_causal_lm_for_classification
            
            self.model = load_causal_lm_for_classification(
                model_name=self.model_config.hf_name,
                num_labels=self.dataset_config.num_labels,
                id2label=self.dataset_config.id2label,
                label2id=self.dataset_config.label2id,
                pad_token_id=self.tokenizer.pad_token_id,
                device_map="auto",
                trust_remote_code=True,
                use_flash_attn=self.model_config.use_flash_attn,
            )
        else:
            # Standard model with native sequence classification support
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.hf_name,
                num_labels=self.dataset_config.num_labels,
                id2label=self.dataset_config.id2label,
                label2id=self.dataset_config.label2id,
                pad_token_id=self.tokenizer.pad_token_id,
                device_map="auto",
                trust_remote_code=True,
            )
        
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        
        self._log_model_params()
    
    def _log_model_params(self):
        """Log model parameter counts."""
        try:
            total_params = self.model.num_parameters()
        except Exception:
            total_params = sum(p.numel() for p in self.model.parameters())
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0
        
        self.log(f"   ✓ Model loaded")
        self.log(f"   📊 Total parameters: {total_params:,}")
        self.log(f"   📊 Trainable parameters: {trainable_params:,} ({trainable_pct:.4f}%)")
        
        self.results["total_parameters"] = total_params
        self.results["trainable_parameters"] = trainable_params
        self.results["trainable_percentage"] = trainable_pct
    
    @abstractmethod
    def _apply_method_specific_setup(self):
        """Apply method-specific setup (e.g., LoRA, BitFit, etc.)."""
        pass
    
    def _tokenize_dataset(self):
        """Tokenize the dataset."""
        self.log(f"\n🔧 Tokenizing dataset...")
        
        # Some models (like Hymba) require fixed-length sequences
        # For these, we pad during tokenization rather than batch collation
        padding_strategy = "max_length" if self.model_config.pad_to_max_length else False
        if self.model_config.pad_to_max_length:
            self.log(f"   ℹ️  Using fixed-length padding (max_length={self.training_config.max_length})")
        
        def tokenize_function(examples):
            # Handle single text or text pair
            if self.dataset_config.text_column_2:
                enc = self.tokenizer(
                    examples[self.dataset_config.text_column],
                    examples[self.dataset_config.text_column_2],
                    truncation=True,
                    padding=padding_strategy,
                    max_length=self.training_config.max_length,
                )
            else:
                enc = self.tokenizer(
                    examples[self.dataset_config.text_column],
                    truncation=True,
                    padding=padding_strategy,
                    max_length=self.training_config.max_length,
                )
            
            enc["labels"] = examples[self.dataset_config.label_column]
            return enc
        
        # Determine columns to remove
        columns_to_remove = [self.dataset_config.text_column]
        if self.dataset_config.text_column_2:
            columns_to_remove.append(self.dataset_config.text_column_2)
        if "idx" in self.dataset["train"].column_names:
            columns_to_remove.append("idx")
        # Don't include label_column if it's the same as "labels"
        if self.dataset_config.label_column != "labels" and self.dataset_config.label_column in self.dataset["train"].column_names:
            columns_to_remove.append(self.dataset_config.label_column)
        
        self.tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove,
            desc="Tokenizing",
        )
        
        self.log("   ✓ Dataset tokenized")
    
    def _get_compute_metrics(self):
        """Get the compute_metrics function for the trainer."""
        metric_name = self.dataset_config.metric_name
        
        if metric_name == "matthews_correlation":
            metric = evaluate.load("matthews_correlation")
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                preds = np.argmax(logits, axis=1)
                return metric.compute(predictions=preds, references=labels)
        else:
            # Default to accuracy
            metric = evaluate.load("accuracy")
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                preds = np.argmax(logits, axis=1)
                return metric.compute(predictions=preds, references=labels)
        
        return compute_metrics
    
    def _get_training_args(self) -> TrainingArguments:
        """Get training arguments."""
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        
        args_kwargs = dict(
            output_dir=self.output_dir,
            num_train_epochs=self.training_config.epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            eval_steps=self.training_config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=self.dataset_config.metric_for_best_model,
            greater_is_better=self.dataset_config.greater_is_better,
            report_to=None,
            dataloader_drop_last=False,
            bf16=bool(bf16_ok),
            fp16=bool(not bf16_ok),
            gradient_checkpointing=False,
            remove_unused_columns=True,
            ddp_find_unused_parameters=False,
        )
        
        return TrainingArguments(**args_kwargs)
    
    def _setup_trainer(self):
        """Setup the Hugging Face Trainer."""
        self.log(f"\n⚙️  Setting up trainer...")
        
        # Setup NVML callback with improved sampling during training
        self.nvml_callback = CheckpointNVMLCallback(
            track_torch_peaks=True,
            gpu_index=self.training_config.gpu_index,
            use_background_sampling=self.training_config.nvml_use_background_sampling,
            sample_interval_ms=self.training_config.nvml_sample_interval_ms,
            sample_every_n_steps=self.training_config.nvml_sample_every_n_steps,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = self._get_training_args()
        
        # Get train and eval splits
        train_split = "train"
        eval_split = "validation" if "validation" in self.tokenized_dataset else "test"
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset[train_split],
            eval_dataset=self.tokenized_dataset[eval_split],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._get_compute_metrics(),
            callbacks=[self.nvml_callback],
        )
        
        self.log("   ✓ Trainer configured")
    
    def run(self) -> Dict[str, Any]:
        """Run the full training pipeline."""
        self.setup()
        
        # Pre-training evaluation
        self.log(f"\n📊 Pre-training evaluation (zero-shot)...")
        pre_eval = self.trainer.evaluate()
        
        # Get the primary metric
        metric_key = self.dataset_config.metric_for_best_model
        zs_metric = float(pre_eval.get(metric_key, 0.0))
        self.log(f"   Zero-shot {self.dataset_config.metric_name}: {zs_metric:.4f}")
        
        # Training
        self.log(f"\n🚀 Starting {self.method_name} fine-tuning...")
        start_time = time.time()
        train_result = self.trainer.train()
        self.training_time_seconds = time.time() - start_time
        
        final_train_loss = getattr(train_result, "training_loss", None)
        self.log("   ✅ Fine-tuning completed!")
        self.log(f"   ⏱️  Training time: {self.training_time_seconds/60:.1f} minutes")
        if final_train_loss is not None:
            self.log(f"   📉 Final training loss: {final_train_loss:.4f}")
        
        # Post-training evaluation
        self.log(f"\n📊 Post-training evaluation...")
        post_eval = self.trainer.evaluate()
        ft_metric = float(post_eval.get(metric_key, 0.0))
        improvement = ft_metric - zs_metric
        self.log(f"   Fine-tuned {self.dataset_config.metric_name}: {ft_metric:.4f}")
        self.log(f"   📈 Improvement: +{improvement:.4f}")
        
        # Collect NVML stats - use the improved sampling
        avg_power_w = None
        avg_vram_used_mb = None
        avg_peak_alloc_mb = None
        max_vram_used_mb = None
        energy_wh = None
        energy_source = None
        num_samples = 0
        
        if hasattr(self.nvml_callback, "summary"):
            s = self.nvml_callback.summary or {}
            
            # Get best power estimate (continuous > step > checkpoint)
            avg_power_w = s.get("avg_power_watts")
            energy_wh = s.get("estimated_energy_wh")
            energy_source = s.get("energy_source", "unknown")
            
            # Get VRAM stats from the best source
            if "continuous_sampling" in s:
                avg_vram_used_mb = s["continuous_sampling"].get("avg_vram_used_mb")
                max_vram_used_mb = s["continuous_sampling"].get("max_vram_used_mb")
                num_samples = s["continuous_sampling"].get("num_samples", 0)
            elif "step_sampling" in s:
                avg_vram_used_mb = s["step_sampling"].get("avg_vram_used_mb")
                max_vram_used_mb = s["step_sampling"].get("max_vram_used_mb")
                num_samples = s["step_sampling"].get("num_samples", 0)
            elif "checkpoint_sampling" in s:
                avg_vram_used_mb = s["checkpoint_sampling"].get("avg_vram_used_mb")
                num_samples = s["checkpoint_sampling"].get("num_samples", 0)
            
            # Peak allocator stats
            if "peak_allocator_mb" in s:
                avg_peak_alloc_mb = s["peak_allocator_mb"].get("avg")
        
        # Calculate SAM metrics
        sam_metrics = compute_sam_metrics(ft_metric, energy_wh)
        
        # Save model
        self.log(f"\n💾 Saving fine-tuned model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        self.log(f"   ✓ Model saved to: {self.output_dir}")
        
        # Compile results
        self.results.update({
            "method": self.method_name,
            "model_name": self.model_config.hf_name,
            "dataset": self.dataset_config.name,
            "metric_name": self.dataset_config.metric_name,
            "seed": self.training_config.seed,
            "batch_size": self.training_config.batch_size,
            "learning_rate": self.training_config.learning_rate,
            "epochs": self.training_config.epochs,
            "max_length": self.training_config.max_length,
            f"zero_shot_{self.dataset_config.metric_name}": zs_metric,
            f"fine_tuned_{self.dataset_config.metric_name}": ft_metric,
            "improvement": improvement,
            "training_time_minutes": self.training_time_seconds / 60.0,
            "final_training_loss": float(final_train_loss) if final_train_loss else None,
            
            # NVML metrics (from improved sampling during training)
            "avg_gpu_power_watts": avg_power_w,
            "avg_gpu_vram_used_mb": avg_vram_used_mb,
            "max_gpu_vram_used_mb": max_vram_used_mb,
            "avg_peak_allocator_mb": avg_peak_alloc_mb,
            "num_power_samples": num_samples,
            "estimated_energy_Wh": energy_wh,
            "energy_measurement_source": energy_source,
            
            # SAM metrics
            **sam_metrics,
        })
        
        # Save results to JSON
        results_path = f"{self.output_dir}/benchmark_results_{self.method_name}.json"
        os.makedirs(self.output_dir, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=4)
        
        self.log(f"   ✓ Results saved to: {results_path}")
        
        # Print summary
        self._print_summary(zs_metric, ft_metric, improvement, avg_power_w, 
                          avg_vram_used_mb, max_vram_used_mb, energy_wh, 
                          energy_source, sam_metrics, num_samples)
        
        return self.results
    
    def _print_summary(self, zs_metric, ft_metric, improvement, avg_power_w,
                      avg_vram_used_mb, max_vram_used_mb, energy_wh, 
                      energy_source, sam_metrics, num_samples):
        """Print training summary."""
        self.log(f"\n{'='*70}")
        self.log(f"🏁 {self.method_name.upper()} FINE-TUNING SUMMARY")
        self.log(f"{'='*70}")
        self.log(f"✅ Training completed successfully!")
        self.log(f"📊 Results:")
        self.log(f"   • Zero-shot {self.dataset_config.metric_name}: {zs_metric:.4f}")
        self.log(f"   • Fine-tuned {self.dataset_config.metric_name}: {ft_metric:.4f}")
        self.log(f"   • Improvement: +{improvement:.4f}")
        self.log(f"   • Training time: {self.training_time_seconds/60:.1f} minutes")
        self.log(f"   • Trainable params: {self.results.get('trainable_parameters', 'N/A'):,}")
        
        self.log(f"\n⚡ Energy Metrics:")
        if avg_power_w is not None:
            self.log(f"   • Avg power: {avg_power_w:.2f} W ({num_samples} samples)")
        if avg_vram_used_mb is not None:
            self.log(f"   • Avg VRAM: {avg_vram_used_mb:.0f} MiB")
        if max_vram_used_mb is not None:
            self.log(f"   • Max VRAM: {max_vram_used_mb:.0f} MiB")
        if energy_wh is not None:
            self.log(f"   • Estimated energy: {energy_wh:.4f} Wh")
            if energy_source:
                self.log(f"   • Measurement source: {energy_source}")
        
        self.log(f"\n📈 SAM Metrics:")
        self.log(format_sam_results(sam_metrics))
        
        self.log(f"\n💾 Model saved to: {self.output_dir}")

