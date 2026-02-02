#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom model wrappers for models that don't natively support sequence classification.

This module provides wrappers that add classification heads on top of CausalLM models,
enabling their use for sequence classification tasks like GLUE benchmarks.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


@dataclass
class SequenceClassifierOutput:
    """Output for sequence classification models."""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class CausalLMForSequenceClassification(nn.Module):
    """
    Wraps a CausalLM model with a classification head for sequence classification.
    
    This is useful for models like Hymba that don't have native 
    AutoModelForSequenceClassification support.
    
    The classification approach:
    1. Run the CausalLM to get hidden states
    2. Pool the last non-padding token's hidden state (GPT-style)
    3. Pass through a classification head
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_labels: int,
        pad_token_id: Optional[int] = None,
        id2label: Optional[dict] = None,
        label2id: Optional[dict] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        
        # Get hidden size from config
        self.config = base_model.config
        if hasattr(self.config, "hidden_size"):
            hidden_size = self.config.hidden_size
        elif hasattr(self.config, "d_model"):
            hidden_size = self.config.d_model
        elif hasattr(self.config, "n_embd"):
            hidden_size = self.config.n_embd
        else:
            raise ValueError("Cannot determine hidden size from model config")
        
        # Detect if model has memory tokens (like Hymba)
        # These models prepend memory tokens internally, which affects sequence length
        self.num_memory_tokens = getattr(self.config, "num_memory_tokens", 0)
        self.has_memory_tokens = self.num_memory_tokens > 0
        
        if self.has_memory_tokens:
            print(f"   ℹ️  Model has {self.num_memory_tokens} memory tokens (hidden states will be offset)")
        
        # Store label mappings in config for compatibility
        self.config.num_labels = num_labels
        self.config.id2label = id2label or {i: str(i) for i in range(num_labels)}
        self.config.label2id = label2id or {str(i): i for i in range(num_labels)}
        self.config.pad_token_id = pad_token_id
        
        # Classification head
        self.score = nn.Linear(hidden_size, num_labels, bias=False)
        
        # Disable caching for training
        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False
    
    def _get_last_non_padding_token_idx(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the index of the last non-padding token for each sequence.
        
        Handles both left-padding and right-padding:
        - Right-padding: [token1, token2, ..., tokenN, PAD, PAD] -> want index N-1
        - Left-padding: [PAD, PAD, token1, token2, ..., tokenN] -> want last index (seq_len - 1)
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        if attention_mask is not None:
            # Check if this is left-padded or right-padded by looking at first token
            # Left-padded: first position is often 0 (padding)
            # Right-padded: last positions are often 0 (padding)
            
            # For each sequence, find the last position where attention_mask is 1
            # This works for both left and right padding
            
            # Create position indices
            positions = torch.arange(seq_len, device=attention_mask.device).expand(batch_size, -1)
            
            # Mask out padding positions with -1, then take max to find last valid position
            masked_positions = positions * attention_mask + (attention_mask - 1)  # padding gets -1
            sequence_lengths = masked_positions.max(dim=1).values
        elif self.pad_token_id is not None:
            # Create a mask where non-padding tokens are 1
            non_pad_mask = (input_ids != self.pad_token_id).long()
            positions = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
            masked_positions = positions * non_pad_mask + (non_pad_mask - 1)
            sequence_lengths = masked_positions.max(dim=1).values
        else:
            # Assume all positions are valid, use last position
            sequence_lengths = torch.full(
                (batch_size,), 
                seq_len - 1, 
                device=input_ids.device, 
                dtype=torch.long
            )
        
        # Clamp to valid range
        sequence_lengths = sequence_lengths.clamp(min=0, max=seq_len - 1)
        
        return sequence_lengths
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[SequenceClassifierOutputWithPast, Tuple]:
        """
        Forward pass for sequence classification.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Classification labels
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dataclass or tuple
            **kwargs: Additional arguments passed to the base model
        
        Returns:
            SequenceClassifierOutputWithPast with loss and logits
        """
        # Filter out kwargs that the base model might not support
        # Some models (like Hymba) have custom forward signatures
        unsupported_kwargs = [
            'num_items_in_batch',  # Added by newer Transformers trainers
            'cache_position',
            'use_cache',
            'past_key_values',
        ]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_kwargs}
        
        batch_size, seq_len = input_ids.shape
        
        # For models with memory tokens (like Hymba), don't pass attention_mask
        # The model handles attention masking internally after adding memory tokens
        # Passing external attention_mask causes size mismatches
        model_attention_mask = None if self.has_memory_tokens else attention_mask
        
        # Forward through base model
        try:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=model_attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,  # Disable cache to avoid state mismatches
                **filtered_kwargs,
            )
        except TypeError as e:
            # If still failing, try minimal args only
            if "unexpected keyword argument" in str(e):
                try:
                    outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=model_attention_mask,
                        output_hidden_states=True,
                    )
                except TypeError:
                    # Last resort: absolute minimal call
                    outputs = self.base_model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                    )
            else:
                raise
        
        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # (batch_size, extended_seq_len, hidden_size)
        
        # For models with memory tokens, the hidden states have prepended memory token states
        # We need to offset our sequence indexing to account for this
        # hidden_states shape: [batch, num_memory_tokens + seq_len, hidden]
        memory_offset = self.num_memory_tokens
        
        # Get the last non-padding token's hidden state for each sequence
        # sequence_lengths gives us the index in the ORIGINAL sequence (0 to seq_len-1)
        sequence_lengths = self._get_last_non_padding_token_idx(input_ids, attention_mask)
        
        # Adjust for memory tokens: add offset to get correct position in hidden_states
        adjusted_positions = sequence_lengths + memory_offset
        
        # Gather the hidden states at the last valid position
        pooled_output = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            adjusted_positions,
        ]  # (batch_size, hidden_size)
        
        # Classification
        logits = self.score(pooled_output)  # (batch_size, num_labels)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=getattr(outputs, "attentions", None),
        )
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing on the base model."""
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(**kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the base model."""
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()
    
    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def get_input_embeddings(self):
        """Get input embeddings from base model."""
        return self.base_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Set input embeddings on base model."""
        self.base_model.set_input_embeddings(value)
    
    def parameters(self, recurse=True):
        """Return all parameters."""
        return super().parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True):
        """Return all named parameters."""
        return super().named_parameters(prefix=prefix, recurse=recurse)


def load_causal_lm_for_classification(
    model_name: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
    pad_token_id: int,
    device_map: str = "auto",
    torch_dtype=None,
    quantization_config=None,
    trust_remote_code: bool = True,
    use_flash_attn: bool = False,
) -> CausalLMForSequenceClassification:
    """
    Load a CausalLM model and wrap it for sequence classification.
    
    Args:
        model_name: HuggingFace model name
        num_labels: Number of classification labels
        id2label: ID to label mapping
        label2id: Label to ID mapping
        pad_token_id: Padding token ID
        device_map: Device placement strategy
        torch_dtype: Torch dtype for model weights
        quantization_config: Optional quantization config (e.g., BitsAndBytesConfig)
        trust_remote_code: Whether to trust remote code
        use_flash_attn: Whether to use Flash Attention 2 for faster attention
    
    Returns:
        CausalLMForSequenceClassification wrapper
    """
    # Load base CausalLM
    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }
    
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    
    # Add Flash Attention 2 support
    if use_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
        # Flash Attention 2 requires bfloat16 or float16
        if torch_dtype is None:
            load_kwargs["torch_dtype"] = torch.bfloat16
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    
    # Wrap with classification head
    model = CausalLMForSequenceClassification(
        base_model=base_model,
        num_labels=num_labels,
        pad_token_id=pad_token_id,
        id2label=id2label,
        label2id=label2id,
    )
    
    return model


