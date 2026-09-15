#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA (Quantized LoRA) training method - 4-bit quantized base model with LoRA.

For standard transformer models this quantizes on-the-fly via BitsAndBytesConfig.
For Mamba-1 (and similar SSM models whose kernels break with on-the-fly BnB
quantization), we instead load a **GPTQ pre-quantized** checkpoint from disk
and apply standard LoRA on top.

Pre-quantize Mamba with:
    python quantize_mamba.py          # requires: pip install auto-gptq optimum
"""

import os

# ── Fix auto-gptq / optimum compatibility (must run before transformers) ──
try:
    import auto_gptq as _ag
    if not hasattr(_ag, "QuantizeConfig"):
        if hasattr(_ag, "BaseQuantizeConfig"):
            _ag.QuantizeConfig = _ag.BaseQuantizeConfig
except ImportError:
    pass  # auto-gptq not installed – only needed for GPTQ path
# ──────────────────────────────────────────────────────────────────────────

import torch
from transformers import (
    AutoModelForCausalLM,
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

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _has_prequantized_model(self) -> bool:
        """Check whether a pre-quantized checkpoint is available on disk."""
        path = getattr(self.model_config, "quantized_model_path", None)
        return path is not None and os.path.isdir(path)

    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        """
        Load the model.

        • If a GPTQ pre-quantized checkpoint exists (Mamba, etc.) → load it
          and wrap with the classification head.
        • Otherwise → standard on-the-fly BnB 4-bit quantisation path.
        • mamba_ssm models (Mamba2) are blocked — their fused CUDA kernels are
          incompatible with BitsAndBytes quantization.
        """
        if self.model_config.requires_mamba_ssm and not self._has_prequantized_model():
            raise NotImplementedError(
                f"QLoRA is not supported for mamba_ssm models ({self.model_config.name}). "
                "mamba_ssm's fused CUDA kernels are incompatible with BitsAndBytes quantization. "
                "Use LoRA or LoRA+ instead."
            )
        if self._has_prequantized_model():
            self._load_prequantized_model()
        else:
            self._load_model_online_quantization()

    # ---- path A: GPTQ pre-quantized (Mamba, etc.) --------------------
    def _load_prequantized_model(self):
        """Load a GPTQ-quantized model that was saved to disk."""
        quant_path = self.model_config.quantized_model_path
        self.log(f"\n🧠 Loading GPTQ pre-quantized model from {quant_path} …")

        # The quantization_config is embedded in config.json — transformers
        # auto-detects GPTQ and loads via auto-gptq; no extra config needed.
        base_model = AutoModelForCausalLM.from_pretrained(
            quant_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Prepare for k-bit training (freeze quantised weights, cast
        # layer-norms to fp32, enable input-embedding grads, etc.)
        base_model = prepare_model_for_kbit_training(base_model)

        if self.model_config.requires_custom_head:
            from utils.custom_models import CausalLMForSequenceClassification

            self.model = CausalLMForSequenceClassification(
                base_model=base_model,
                num_labels=self.dataset_config.num_labels,
                pad_token_id=self.tokenizer.pad_token_id,
                id2label=self.dataset_config.id2label,
                label2id=self.dataset_config.label2id,
            )
        else:
            self.model = base_model

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        self.log(f"   ✓ Pre-quantized model loaded (GPTQ 4-bit)")
        self._log_model_params()

    # ---- path B: on-the-fly BnB quantization (default) ---------------
    def _load_model_online_quantization(self):
        """Load the model with on-the-fly 4-bit BnB quantization."""
        self.log(f"\n🧠 Loading base model in 4-bit (BnB) …")

        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        bnb_kwargs = dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
        )
        if self.model_config.quantization_skip_modules:
            bnb_kwargs["llm_int8_skip_modules"] = (
                self.model_config.quantization_skip_modules
            )
        bnb_config = BitsAndBytesConfig(**bnb_kwargs)

        if self.model_config.requires_custom_head:
            from utils.custom_models import load_causal_lm_for_classification

            self.model = load_causal_lm_for_classification(
                model_name=self.model_config.hf_name,
                num_labels=self.dataset_config.num_labels,
                id2label=self.dataset_config.id2label,
                label2id=self.dataset_config.label2id,
                pad_token_id=self.tokenizer.pad_token_id,
                device_map="auto",
                torch_dtype=self.model_config.torch_dtype,
                quantization_config=bnb_config,
                trust_remote_code=True,
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

        self.model = prepare_model_for_kbit_training(self.model)

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        self.log(f"   ✓ Base model loaded (4-bit NF4)")
        self._log_model_params()

    # ------------------------------------------------------------------
    # LoRA setup
    # ------------------------------------------------------------------
    def _get_target_modules(self):
        """
        Get target modules for LoRA.
        Re-uses the auto-discovery logic from the LoRA trainer so that
        modules are verified against the (possibly wrapped) model.
        """
        target_modules = self.model_config.target_modules

        if self.model_config.requires_custom_head:
            from utils.module_discovery import find_target_modules_for_lora

            base_model = getattr(self.model, "base_model", self.model)

            all_module_names = set()
            for name, _ in base_model.named_modules():
                all_module_names.update(name.split("."))

            if not any(m in all_module_names for m in target_modules):
                self.log(
                    f"   ⚠️  Configured target modules {target_modules} not found"
                )
                self.log(f"   🔍 Auto-discovering target modules …")
                discovered = find_target_modules_for_lora(base_model)
                if discovered:
                    target_modules = discovered
                    self.log(f"   ✓ Discovered modules: {target_modules}")
                else:
                    self.log(f"   ⚠️  Using 'all-linear' fallback")
                    target_modules = "all-linear"

        return target_modules

    def _apply_method_specific_setup(self):
        """Apply LoRA adapters on top of the (quantized) model."""
        self.log(f"\n🔧 Configuring LoRA for QLoRA …")

        target_modules = self._get_target_modules()

        lora_kwargs = dict(
            task_type=TaskType.SEQ_CLS,
            r=self.training_config.lora_r,
            lora_alpha=self.training_config.lora_alpha,
            lora_dropout=0.05,  # Lower dropout for QLoRA
            target_modules=target_modules,
            bias="none",
        )
        if self.model_config.modules_to_save:
            lora_kwargs["modules_to_save"] = self.model_config.modules_to_save
        lora_config = LoraConfig(**lora_kwargs)

        self.model = get_peft_model(self.model, lora_config)

        self._log_model_params()
        mode = "GPTQ" if self._has_prequantized_model() else "BnB 4-bit"
        self.log(f"   ✓ QLoRA applied ({mode} + LoRA)")
        self.log(
            f"   • r={self.training_config.lora_r}, alpha={self.training_config.lora_alpha}"
        )
        self.log(f"   • Target modules: {target_modules}")

    # ------------------------------------------------------------------
    # training args & trainer
    # ------------------------------------------------------------------
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
            gradient_checkpointing=False,
            remove_unused_columns=True,
            ddp_find_unused_parameters=False,
            save_safetensors=not (self.model_config.requires_custom_head or self.model_config.requires_mamba_ssm),
        )

        return TrainingArguments(**args_kwargs)

    def _setup_trainer(self):
        """Setup trainer with appropriate optimizer for QLoRA."""
        from transformers import DataCollatorWithPadding
        from utils.nvml_callback import CheckpointNVMLCallback

        self.log(f"\n⚙️  Setting up QLoRA trainer …")

        self.nvml_callback = CheckpointNVMLCallback(
            track_torch_peaks=True,
            gpu_index=self.training_config.gpu_index,
            use_background_sampling=self.training_config.nvml_use_background_sampling,
            sample_interval_ms=self.training_config.nvml_sample_interval_ms,
            sample_every_n_steps=self.training_config.nvml_sample_every_n_steps,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = self._get_training_args()

        # ---- optimizer ---------------------------------------------------
        # For BnB path: use paged 8-bit AdamW (classic QLoRA).
        # For GPTQ path: use standard AdamW (BnB paged optimizer is optional
        #   but still works if bitsandbytes is installed).
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.PagedAdamW8bit(
                self.model.parameters(),
                lr=training_args.learning_rate,
                weight_decay=training_args.weight_decay,
            )
            self.log("   • Optimizer: PagedAdamW8bit")
        except ImportError:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=training_args.learning_rate,
                weight_decay=training_args.weight_decay,
            )
            self.log("   • Optimizer: AdamW (bitsandbytes not available)")

        # ---- scheduler ---------------------------------------------------
        train_split = "train"
        updates_per_epoch = (
            len(self.tokenized_dataset[train_split])
            // training_args.per_device_train_batch_size
        )
        num_training_steps = max(1, updates_per_epoch) * int(
            training_args.num_train_epochs
        )
        num_warmup_steps = int(num_training_steps * training_args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

        eval_split = (
            "validation" if "validation" in self.tokenized_dataset else "test"
        )

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
