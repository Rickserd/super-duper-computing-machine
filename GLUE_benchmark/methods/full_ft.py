#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full fine-tuning - All parameters are trainable.
"""

import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments

from .base import BaseTrainer


class FullFTTrainer(BaseTrainer):
    """Full fine-tuning: All parameters are trainable."""
    
    method_name = "full_ft"
    
    def _load_model(self):
        """Load the base model without device_map for full fine-tuning."""
        self.log(f"\n🧠 Loading base model (full fine-tune)...")
        
        load_dtype = self._get_load_dtype()
        
        if self.model_config.requires_mamba_ssm:
            from utils.custom_models import load_mamba2_for_classification
            self.model = load_mamba2_for_classification(
                model_name=self.model_config.hf_name,
                num_labels=self.dataset_config.num_labels,
                id2label=self.dataset_config.id2label,
                label2id=self.dataset_config.label2id,
                pad_token_id=self.tokenizer.pad_token_id,
                torch_dtype=load_dtype,
            )
        elif self.model_config.requires_custom_head:
            from utils.custom_models import load_causal_lm_for_classification
            self.model = load_causal_lm_for_classification(
                model_name=self.model_config.hf_name,
                num_labels=self.dataset_config.num_labels,
                id2label=self.dataset_config.id2label,
                label2id=self.dataset_config.label2id,
                pad_token_id=self.tokenizer.pad_token_id,
                device_map=None,
                torch_dtype=load_dtype,
                trust_remote_code=True,
            )
        else:
            load_kwargs = dict(
                num_labels=self.dataset_config.num_labels,
                id2label=self.dataset_config.id2label,
                label2id=self.dataset_config.label2id,
                pad_token_id=self.tokenizer.pad_token_id,
                torch_dtype=load_dtype,
                trust_remote_code=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.hf_name,
                **load_kwargs,
            )
        
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        
        # Move to device (mamba_ssm models are already on device)
        if not self.model_config.requires_mamba_ssm:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
        
        self._log_model_params()
    
    def _apply_method_specific_setup(self):
        """No special setup for full fine-tuning - all params already trainable."""
        self.log(f"\n🔧 Full fine-tuning mode (all parameters trainable)")
        self.log("   ✓ All parameters set as trainable")
    
    def _get_training_args(self) -> TrainingArguments:
        """Get training arguments with gradient checkpointing for memory efficiency."""
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        
        # Try to use 8-bit optimizer if available
        optim = "adamw_torch"
        try:
            import bitsandbytes
            optim = "paged_adamw_8bit"
        except ImportError:
            pass
        
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
            dataloader_num_workers=4,
            bf16=bool(bf16_ok),
            fp16=bool(not bf16_ok),
            gradient_checkpointing=False,  # Enable for memory efficiency
            optim=optim,
            remove_unused_columns=True,
            ddp_find_unused_parameters=False,
            save_safetensors=not (self.model_config.requires_custom_head or self.model_config.requires_mamba_ssm),
        )
        
        return TrainingArguments(**args_kwargs)

