#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA+ training method - LoRA with different learning rates for A and B matrices.
"""

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
from transformers.optimization import get_linear_schedule_with_warmup

from .base import BaseTrainer


class LoraPlusTrainer(BaseTrainer):
    """LoRA+: LoRA with different learning rates for A and B matrices."""
    
    method_name = "loraplus"
    
    def _get_target_modules(self):
        """
        Get target modules for LoRA. 
        Auto-discovers modules for custom models if the default ones don't exist.
        """
        target_modules = self.model_config.target_modules
        
        if self.model_config.requires_custom_head:
            from utils.module_discovery import find_target_modules_for_lora
            
            base_model = getattr(self.model, 'base_model', self.model)
            
            all_module_names = set()
            for name, _ in base_model.named_modules():
                parts = name.split(".")
                all_module_names.update(parts)
            
            modules_exist = any(m in all_module_names for m in target_modules)
            
            if not modules_exist:
                self.log(f"   ⚠️  Configured target modules {target_modules} not found in model")
                self.log(f"   🔍 Auto-discovering target modules...")
                
                discovered = find_target_modules_for_lora(base_model)
                if discovered:
                    target_modules = discovered
                    self.log(f"   ✓ Discovered modules: {target_modules}")
                else:
                    self.log(f"   ⚠️  Using 'all-linear' fallback")
                    target_modules = "all-linear"
        
        return target_modules
    
    def _apply_method_specific_setup(self):
        """Apply LoRA adapters to the model."""
        self.log(f"\n🔧 Configuring LoRA+...")
        
        target_modules = self._get_target_modules()
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.training_config.lora_r,
            lora_alpha=self.training_config.lora_alpha,
            lora_dropout=self.training_config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Update parameter counts
        self._log_model_params()
        self.log("   ✓ LoRA+ applied")
        self.log(f"   • r={self.training_config.lora_r}, alpha={self.training_config.lora_alpha}")
        self.log(f"   • LoRA+ ratio: {self.training_config.loraplus_ratio}")
        self.log(f"   • Target modules: {target_modules}")
    
    def _build_loraplus_param_groups(self, lr: float, ratio: int):
        """Build parameter groups with different learning rates for A and B matrices."""
        a_params, b_params, rest = [], [], []
        
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_A" in n:
                a_params.append(p)
            elif "lora_B" in n:
                b_params.append(p)
            else:
                rest.append(p)  # classifier head etc.
        
        groups = [
            {"params": a_params, "lr": lr},
            {"params": b_params, "lr": lr * ratio},
        ]
        if rest:
            groups.append({"params": rest, "lr": lr})
        
        return groups
    
    def _setup_trainer(self):
        """Setup trainer with LoRA+ optimizer."""
        from transformers import DataCollatorWithPadding
        from utils.nvml_callback import CheckpointNVMLCallback
        
        self.log(f"\n⚙️  Setting up LoRA+ trainer...")
        
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
        
        # Build LoRA+ optimizer
        optimizer = None
        scheduler = None
        
        try:
            from peft.optimizers import create_loraplus_optimizer
            try:
                import bitsandbytes as bnb
                opt_cls = bnb.optim.Adam8bit
            except ImportError:
                opt_cls = torch.optim.AdamW
            
            optimizer = create_loraplus_optimizer(
                model=self.model,
                optimizer_cls=opt_cls,
                lr=training_args.learning_rate,
                loraplus_lr_ratio=self.training_config.loraplus_ratio,
                weight_decay=training_args.weight_decay,
            )
            self.log("   ✓ Using PEFT's create_loraplus_optimizer")
        except Exception as e:
            self.log(f"   ⚠️ Falling back to manual LoRA+ param groups: {e}")
            param_groups = self._build_loraplus_param_groups(
                training_args.learning_rate,
                self.training_config.loraplus_ratio,
            )
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.Adam8bit(param_groups, weight_decay=training_args.weight_decay)
            except ImportError:
                optimizer = torch.optim.AdamW(param_groups, weight_decay=training_args.weight_decay)
        
        # Build scheduler
        train_split = "train"
        updates_per_epoch = len(self.tokenized_dataset[train_split]) // training_args.per_device_train_batch_size
        num_training_steps = max(1, updates_per_epoch) * int(training_args.num_train_epochs)
        num_warmup_steps = int(num_training_steps * training_args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        
        # Get train and eval splits
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
            optimizers=(optimizer, scheduler),
        )
        
        self.log("   ✓ LoRA+ trainer configured")

