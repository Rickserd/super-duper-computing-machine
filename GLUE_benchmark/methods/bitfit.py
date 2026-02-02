#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BitFit training method - Only bias parameters are trainable.
"""

from .base import BaseTrainer


class BitFitTrainer(BaseTrainer):
    """BitFit: Only train bias terms, layer norms, and classifier head."""
    
    method_name = "bitfit"
    
    def _apply_method_specific_setup(self):
        """Apply BitFit - freeze all except biases, norms, and classifier."""
        self.log(f"\n🪛 Applying BitFit (bias + norm + classifier head trainable)...")
        
        # First freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Unfreeze classifier head (usually 'score.*' for sequence classification)
        for name, p in self.model.named_parameters():
            if name.startswith("score.") or ".score." in name:
                p.requires_grad = True
            # Also check for 'classifier' naming convention
            if "classifier" in name.lower():
                p.requires_grad = True
        
        def is_bitfit_param(n: str) -> bool:
            """Check if parameter should be trainable in BitFit."""
            n = n.lower()
            return (
                n.endswith(".bias")
                or ".bias" in n
                or "norm.weight" in n
                or "rmsnorm.weight" in n
                or ".ln_" in n
                or "layernorm" in n
            )
        
        # Unfreeze BitFit parameters
        for name, p in self.model.named_parameters():
            if is_bitfit_param(name):
                p.requires_grad = True
        
        # Log updated parameter counts
        self._log_model_params()
        self.log("   ✓ BitFit applied")

