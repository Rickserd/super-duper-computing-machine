#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA (Low-Rank Adaptation) training method.
"""

from peft import LoraConfig, get_peft_model, TaskType

from .base import BaseTrainer


class LoRATrainer(BaseTrainer):
    """LoRA: Low-Rank Adaptation for efficient fine-tuning."""
    
    method_name = "lora"
    
    def _get_target_modules(self):
        """
        Get target modules for LoRA. 
        Auto-discovers modules for custom models if the default ones don't exist.
        """
        target_modules = self.model_config.target_modules
        
        # For custom head models, we need to check if the target modules exist
        # in the base_model (the wrapped CausalLM), not the wrapper
        if self.model_config.requires_custom_head:
            from utils.module_discovery import find_target_modules_for_lora
            
            # Get the base model from the wrapper
            base_model = getattr(self.model, 'base_model', self.model)
            
            # Check if configured modules exist
            all_module_names = set()
            for name, _ in base_model.named_modules():
                parts = name.split(".")
                all_module_names.update(parts)
            
            # Check if any of our target modules exist
            modules_exist = any(m in all_module_names for m in target_modules)
            
            if not modules_exist:
                self.log(f"   ⚠️  Configured target modules {target_modules} not found in model")
                self.log(f"   🔍 Auto-discovering target modules...")
                
                discovered = find_target_modules_for_lora(base_model)
                if discovered:
                    target_modules = discovered
                    self.log(f"   ✓ Discovered modules: {target_modules}")
                else:
                    # Fallback to targeting all linear layers
                    self.log(f"   ⚠️  Using 'all-linear' fallback")
                    target_modules = "all-linear"
        
        return target_modules
    
    def _apply_method_specific_setup(self):
        """Apply LoRA adapters to the model."""
        self.log(f"\n🔧 Configuring LoRA...")
        
        # Get appropriate target modules
        target_modules = self._get_target_modules()
        
        # For custom head models, we apply LoRA to the base_model
        # but the wrapper still works with get_peft_model
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
        self.log("   ✓ LoRA applied")
        self.log(f"   • r={self.training_config.lora_r}, alpha={self.training_config.lora_alpha}")
        self.log(f"   • Target modules: {target_modules}")
