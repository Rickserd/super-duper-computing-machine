#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA (Quantized LoRA) training method - 4-bit quantized base model with LoRA.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from transformers.optimization import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from .base import BaseTrainer


class QLoRATrainer(BaseTrainer):
    """QLoRA: 4-bit quantized base model with LoRA adapters."""
    
    method_name = "qlora"
    
    def _load_model(self):
        """Load the model with 4-bit quantization."""
        self.log(f"\n🧠 Loading base model in 4-bit...")
        
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
        )
        
        if self.model_config.requires_custom_head:
            # Model doesn't support AutoModelForSequenceClassification natively
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
                quantization_config=bnb_config,
                trust_remote_code=True,
                use_flash_attn=self.model_config.use_flash_attn,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.hf_name,
                num_labels=self.dataset_config.num_labels,
                id2label=self.dataset_config.id2label,
                label2id=self.dataset_config.label2id,
                pad_token_id=self.tokenizer.pad_token_id,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        
        self.log(f"   ✓ Base model loaded (4-bit NF4)")
        self._log_model_params()
    
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
        """Apply LoRA adapters on top of quantized model."""
        self.log(f"\n🔧 Configuring LoRA for QLoRA...")
        
        target_modules = self._get_target_modules()
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.training_config.lora_r,
            lora_alpha=self.training_config.lora_alpha,
            lora_dropout=0.05,  # Lower dropout for QLoRA
            target_modules=target_modules,
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Update parameter counts
        self._log_model_params()
        self.log("   ✓ QLoRA applied (4-bit + LoRA)")
        self.log(f"   • r={self.training_config.lora_r}, alpha={self.training_config.lora_alpha}")
        self.log(f"   • Target modules: {target_modules}")
    
    def _get_training_args(self) -> TrainingArguments:
        """Get training arguments with gradient checkpointing for QLoRA."""
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
            gradient_checkpointing=True,  # Important for QLoRA
            remove_unused_columns=True,
            ddp_find_unused_parameters=False,
        )
        
        return TrainingArguments(**args_kwargs)
    
    def _setup_trainer(self):
        """Setup trainer with 8-bit paged optimizer for QLoRA."""
        from transformers import DataCollatorWithPadding
        from utils.nvml_callback import CheckpointNVMLCallback
        import bitsandbytes as bnb
        
        self.log(f"\n⚙️  Setting up QLoRA trainer...")
        
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
        
        # Use paged 8-bit optimizer
        optimizer = bnb.optim.PagedAdamW8bit(
            self.model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )
        
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
        
        self.log("   ✓ QLoRA trainer configured")

