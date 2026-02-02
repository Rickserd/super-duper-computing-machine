"""
LoRA Finetuning Script for LaMP Tasks (LaMP-1 through LaMP-4)
Supports TinyLlama, Flan-T5, Mamba2-1.3B, and Qwen3-1.7B models with Contriever retrieval (k=4)

Usage (single task):
    python finetune_lamp_lora.py \
        --model_type flan-t5 \
        --task LaMP-1 \
        --data_dir ./dataset1 \
        --output_dir ./outputs \
        --num_retrieved 4 \
        --num_epochs 3 \
        --batch_size 4

Usage (all tasks):
    python finetune_lamp_lora.py \
        --model_type flan-t5 \
        --task all \
        --data_dir ./data \
        --output_dir ./outputs \
        --num_retrieved 4 \
        --num_epochs 3
    
    Note: For --task all, data_dir should contain dataset1/, dataset2/, dataset3/, dataset4/ folders

Mamba2-1.3B example:
    python finetune_lamp_lora.py \
        --model_type mamba2 \
        --task lamp1 \
        --data_dir ./dataset1 \
        --output_dir ./outputs \
        --num_epochs 3 \
        --bf16

Qwen3-1.7B example:
    python finetune_lamp_lora.py \
        --model_type qwen3 \
        --task lamp1 \
        --data_dir ./dataset1 \
        --output_dir ./outputs \
        --num_epochs 3 \
        --bf16
"""

import argparse
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import datasets
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PretrainedConfig,
    TrainerCallback,
)
from transformers.modeling_outputs import CausalLMOutput
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

# ---- NVML (power/VRAM) optional imports ----
NVML_OK = False
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetMemoryInfo,
    )
    NVML_OK = True
except Exception:
    try:
        from nvidia_ml_py import (
            nvmlInit,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetPowerUsage,
            nvmlDeviceGetMemoryInfo,
        )
        NVML_OK = True
    except Exception:
        NVML_OK = False

# ---- mamba_ssm imports (required for state-spaces/mamba2-* models) ----
MAMBA_SSM_OK = False
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    MAMBA_SSM_OK = True
except ImportError:
    pass


# =============================================================================
# MAMBA2 CONFIG AND MODEL WRAPPERS (for PEFT compatibility)
# =============================================================================

class Mamba2ConfigWrapper(PretrainedConfig):
    """Wrapper to make mamba_ssm config compatible with HuggingFace/PEFT."""
    model_type = "mamba2"
    
    def __init__(self, mamba_config=None, **kwargs):
        super().__init__(**kwargs)
        if mamba_config is not None:
            # Copy all attributes from mamba_ssm config
            for key, value in vars(mamba_config).items():
                setattr(self, key, value)
        # Ensure required attributes exist (defaults for 1.3B model)
        self.hidden_size = getattr(self, 'd_model', 2048)
        self.num_hidden_layers = getattr(self, 'n_layer', 64)


class Mamba2ForCausalLM(PreTrainedModel):
    """
    Custom wrapper to make Mamba2 compatible with HuggingFace Trainer for causal LM.
    Uses mamba_ssm's MambaLMHeadModel as backbone.
    """
    config_class = Mamba2ConfigWrapper
    base_model_prefix = "backbone"
    
    def __init__(self, config: Mamba2ConfigWrapper, backbone=None):
        super().__init__(config)
        self.backbone = backbone
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # Get logits from Mamba backbone
        # MambaLMHeadModel returns CausalLMOutput with logits
        lm_logits = self.backbone(input_ids).logits  # (batch, seq_len, vocab_size)
        
        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=None,
        )
    
    def get_input_embeddings(self):
        return self.backbone.backbone.embedding
    
    def set_input_embeddings(self, value):
        self.backbone.backbone.embedding = value
    
    def generate(self, input_ids, max_new_tokens=32, **kwargs):
        """Generate text using the mamba_ssm generate method."""
        # mamba_ssm's generate method signature is different
        return self.backbone.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + max_new_tokens,
            **kwargs
        )
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


# =============================================================================
# SAM (SUSTAINABLE AI METRIC) UTILITIES
# =============================================================================

def calculate_sam(accuracy: float, energy_wh: float, exponent: int = 1) -> Optional[float]:
    """
    Calculate SAM (Sustainable AI Metric).
    
    SAM(a) = acc^a × a / log(Wh)
    
    Args:
        accuracy: Performance metric (0.0 to 1.0), e.g., accuracy or MCC
        energy_wh: Energy consumption in Watt-hours
        exponent: The 'a' parameter (typically 1, 2, or 5)
    
    Returns:
        SAM value, or None if energy_wh <= 0 or invalid
    """
    import math
    
    if energy_wh <= 0:
        return None
    
    try:
        log_wh = math.log(energy_wh)
        if log_wh == 0:
            return None
        sam = (accuracy ** exponent) * exponent / log_wh
        return sam
    except (ValueError, ZeroDivisionError):
        return None


def calculate_all_sam_metrics(accuracy: float, energy_wh: float) -> Dict[str, Optional[float]]:
    """
    Calculate SAM metrics for a=1, 2, and 5.
    
    Args:
        accuracy: Performance metric (0.0 to 1.0)
        energy_wh: Energy consumption in Watt-hours
    
    Returns:
        Dictionary with SAM_1, SAM_2, SAM_5 values
    """
    return {
        "SAM_1": calculate_sam(accuracy, energy_wh, exponent=1),
        "SAM_2": calculate_sam(accuracy, energy_wh, exponent=2),
        "SAM_5": calculate_sam(accuracy, energy_wh, exponent=5),
    }


# =============================================================================
# NVML TRAINING STEP CALLBACK (Power/VRAM Monitoring During Compute)
# =============================================================================

class TrainingStepNVMLCallback(TrainerCallback):
    """
    Samples NVML power (W) and VRAM used (MiB) during actual training steps.
    
    This callback samples GPU metrics during forward/backward passes (on_step_end),
    excluding I/O operations like data loading and checkpoint saving.
    Provides accurate power/VRAM measurements for energy consumption calculation.
    """
    
    def __init__(self, gpu_index: int = 0, output_dir: str = None, sample_every_n_steps: int = 1):
        self.gpu_index = gpu_index
        self.output_dir = output_dir
        self.sample_every_n_steps = sample_every_n_steps
        self.nvml_ok = False
        self.handle = None
        
        # Training step samples (compute-only, excludes I/O)
        self.step_power_samples = []
        self.step_vram_samples = []
        self.step_timestamps = []
        
        # Checkpoint samples (for comparison)
        self.checkpoint_samples = []
        
        # Timing
        self.train_start_time = None
        self.train_end_time = None
        self.total_compute_time_secs = 0.0
        self.step_start_time = None
        
        # Summary
        self.summary = {}

    def _nvml_read(self) -> Tuple[float, float, float]:
        """Read power (W), VRAM used (MiB), VRAM total (MiB)."""
        if self.handle is None:
            self.handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
        power_w = nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        mem = nvmlDeviceGetMemoryInfo(self.handle)
        used_mb = float(mem.used) / (1024 ** 2)
        total_mb = float(mem.total) / (1024 ** 2)
        return power_w, used_mb, total_mb

    def on_train_begin(self, args, state, control, **kwargs):
        self.nvml_ok = False
        if NVML_OK:
            try:
                nvmlInit()
                self.handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
                self.nvml_ok = True
            except Exception as e:
                print(f"NVML initialization failed: {e}")
                self.nvml_ok = False
        
        self.train_start_time = time.time()
        self.step_power_samples = []
        self.step_vram_samples = []
        self.step_timestamps = []
        self.checkpoint_samples = []
        self.total_compute_time_secs = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        """Mark the start of a training step for compute time tracking."""
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """
        Sample GPU metrics at the end of each training step.
        This captures metrics during actual compute (forward + backward pass),
        excluding data loading and other I/O operations.
        """
        # Track compute time
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.total_compute_time_secs += step_time
        
        # Sample NVML metrics
        if self.nvml_ok and state.global_step % self.sample_every_n_steps == 0:
            try:
                power_w, used_mb, total_mb = self._nvml_read()
                timestamp = time.time()
                
                self.step_power_samples.append(float(power_w))
                self.step_vram_samples.append(float(used_mb))
                self.step_timestamps.append(timestamp)
                
            except Exception:
                pass

    def on_save(self, args, state, control, **kwargs):
        """Sample at checkpoints for comparison (includes I/O context)."""
        if self.nvml_ok:
            try:
                power_w, used_mb, total_mb = self._nvml_read()
                self.checkpoint_samples.append({
                    "global_step": int(state.global_step),
                    "power_watts": float(power_w),
                    "vram_used_mb": float(used_mb),
                    "vram_total_mb": float(total_mb),
                    "timestamp": time.time(),
                })
            except Exception:
                pass

    def on_train_end(self, args, state, control, **kwargs):
        """Compute summary statistics and save results."""
        import statistics
        
        self.train_end_time = time.time()
        total_training_time = self.train_end_time - self.train_start_time if self.train_start_time else 0
        
        # Compute statistics from training step samples
        avg_power = None
        max_power = None
        min_power = None
        avg_vram = None
        max_vram = None
        
        if self.step_power_samples:
            avg_power = float(statistics.mean(self.step_power_samples))
            max_power = float(max(self.step_power_samples))
            min_power = float(min(self.step_power_samples))
        
        if self.step_vram_samples:
            avg_vram = float(statistics.mean(self.step_vram_samples))
            max_vram = float(max(self.step_vram_samples))
        
        # Calculate energy consumption (Wh) = avg_power (W) × time (h)
        energy_wh = None
        if avg_power is not None and total_training_time > 0:
            energy_wh = avg_power * (total_training_time / 3600.0)
        
        self.summary = {
            # Power metrics (from training steps only)
            "avg_power_watts": avg_power,
            "max_power_watts": max_power,
            "min_power_watts": min_power,
            
            # VRAM metrics (from training steps only)
            "avg_vram_used_mb": avg_vram,
            "max_vram_used_mb": max_vram,
            
            # Energy consumption
            "energy_consumption_wh": energy_wh,
            
            # Timing
            "total_training_time_secs": total_training_time,
            "total_compute_time_secs": self.total_compute_time_secs,
            "compute_ratio": self.total_compute_time_secs / total_training_time if total_training_time > 0 else None,
            
            # Sample counts
            "num_step_samples": len(self.step_power_samples),
            "num_checkpoint_samples": len(self.checkpoint_samples),
        }
        
        # Save detailed timeseries and summary
        try:
            out_dir = self.output_dir or args.output_dir
            os.makedirs(out_dir, exist_ok=True)
            
            data = {
                "summary": self.summary,
                "training_step_samples": {
                    "power_watts": self.step_power_samples[-100:] if len(self.step_power_samples) > 100 else self.step_power_samples,  # Last 100 samples
                    "vram_used_mb": self.step_vram_samples[-100:] if len(self.step_vram_samples) > 100 else self.step_vram_samples,
                    "total_samples": len(self.step_power_samples),
                },
                "checkpoint_samples": self.checkpoint_samples,
            }
            
            path = f"{out_dir}/nvml_metrics.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save NVML metrics: {e}")
    
    def get_energy_wh(self) -> Optional[float]:
        """Get energy consumption in Watt-hours."""
        return self.summary.get("energy_consumption_wh")
    
    def get_avg_power(self) -> Optional[float]:
        """Get average power in Watts."""
        return self.summary.get("avg_power_watts")
    
    def get_avg_vram_mb(self) -> Optional[float]:
        """Get average VRAM usage in MiB."""
        return self.summary.get("avg_vram_used_mb")


# Legacy callback for backward compatibility
class CheckpointNVMLCallback(TrainerCallback):
    """Samples NVML power (W) and VRAM used (MiB) at each checkpoint save."""
    
    def __init__(self, gpu_index: int = 0, output_dir: str = None):
        self.gpu_index = gpu_index
        self.output_dir = output_dir
        self.nvml_ok = False
        self.samples_power_w = []
        self.samples_vram_used_mb = []
        self.timeseries = []
        self.summary = {}

    def _nvml_read(self):
        h = nvmlDeviceGetHandleByIndex(self.gpu_index)
        power_w = nvmlDeviceGetPowerUsage(h) / 1000.0
        mem = nvmlDeviceGetMemoryInfo(h)
        used_mb = float(mem.used) / (1024 ** 2)
        total_mb = float(mem.total) / (1024 ** 2)
        return power_w, used_mb, total_mb

    def on_train_begin(self, args, state, control, **kwargs):
        self.nvml_ok = False
        if NVML_OK:
            try:
                nvmlInit()
                self.nvml_ok = True
            except Exception:
                self.nvml_ok = False

    def on_save(self, args, state, control, **kwargs):
        if self.nvml_ok:
            try:
                power_w, used_mb, total_mb = self._nvml_read()
                self.samples_power_w.append(float(power_w))
                self.samples_vram_used_mb.append(float(used_mb))
                self.timeseries.append({
                    "global_step": int(state.global_step),
                    "power_watts": float(power_w),
                    "vram_used_mb": float(used_mb),
                    "vram_total_mb": float(total_mb),
                })
                out_dir = self.output_dir or args.output_dir
                os.makedirs(out_dir, exist_ok=True)
                with open(f"{out_dir}/power_vram_timeseries.json", "w") as f:
                    json.dump({"samples": self.timeseries}, f, indent=2)
            except Exception:
                pass

    def on_train_end(self, args, state, control, **kwargs):
        import statistics
        self.summary = {
            "avg_power_watts_over_checkpoints": (
                float(statistics.mean(self.samples_power_w)) if self.samples_power_w else None
            ),
            "avg_vram_used_mb_over_checkpoints": (
                float(statistics.mean(self.samples_vram_used_mb)) if self.samples_vram_used_mb else None
            ),
            "num_checkpoints_sampled": len(self.samples_power_w),
        }
        try:
            out_dir = self.output_dir or args.output_dir
            os.makedirs(out_dir, exist_ok=True)
            path = f"{out_dir}/power_vram_timeseries.json"
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {"samples": []}
            data["summary"] = self.summary
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


# =============================================================================
# CONTRIEVER RETRIEVAL MODULE
# =============================================================================

def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling for sentence embeddings."""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def batchify(lst: List, batch_size: int) -> List[List]:
    """Split list into batches."""
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]


class ContrieverRetriever:
    """Contriever-based retrieval for user profile items."""
    
    def __init__(self, device: str = "cuda:0", checkpoint: str = "facebook/contriever"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(device)
        self.model.eval()
    
    @torch.no_grad()
    def retrieve_top_k(
        self, 
        corpus: List[str], 
        profile: List[Dict], 
        query: str, 
        k: int,
        batch_size: int = 4
    ) -> List[Dict]:
        """Retrieve top-k profile items based on query similarity."""
        if len(profile) == 0:
            return []
        
        k = min(k, len(profile))
        
        # Encode query
        query_tokens = self.tokenizer(
            [query], padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)
        output_query = self.model(**query_tokens)
        output_query = mean_pooling(output_query.last_hidden_state, query_tokens['attention_mask'])
        
        # Encode corpus in batches
        scores = []
        batched_corpus = batchify(corpus, batch_size)
        for batch in batched_corpus:
            tokens_batch = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
            outputs_batch = self.model(**tokens_batch)
            outputs_batch = mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
            temp_scores = output_query.squeeze() @ outputs_batch.T
            if temp_scores.dim() == 0:
                scores.append(temp_scores.item())
            else:
                scores.extend(temp_scores.tolist())
        
        topk_values, topk_indices = torch.topk(torch.tensor(scores), k)
        return [profile[m] for m in topk_indices.tolist()]
    
    def to_cpu(self):
        """Move model to CPU to free GPU memory."""
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()
    
    def to_gpu(self, device: str = "cuda:0"):
        """Move model back to GPU."""
        self.model = self.model.to(device)
        self.device = device


# =============================================================================
# QUERY/CORPUS MAKERS FOR EACH TASK
# =============================================================================

def extract_strings_between_quotes(input_string: str) -> List[str]:
    """Extract strings between quotes."""
    output_list = []
    inside_quotes = False
    current_string = ''
    
    for char in input_string:
        if char == '"' and not inside_quotes:
            inside_quotes = True
        elif char == '"' and inside_quotes:
            inside_quotes = False
            output_list.append(current_string)
            current_string = ''
        elif inside_quotes:
            current_string += char
    
    return output_list


def extract_after_keyword(input_string: str, keyword: str) -> Optional[str]:
    """Extract text after a keyword."""
    index = input_string.find(keyword)
    if index == -1:
        return input_string
    return input_string[index + len(keyword):].strip()


def lamp1_query_corpus(inp: str, profile: List[Dict]) -> Tuple[List[str], str]:
    """LaMP-1: Citation classification.
    Profile has: title, abstract, id
    """
    corpus = [f'{x["title"]} {x.get("abstract", "")}' for x in profile]
    extracted = extract_strings_between_quotes(inp)
    query = f'{extracted[1]} {extracted[2]}' if len(extracted) >= 3 else inp
    return corpus, query


def lamp2_query_corpus(inp: str, profile: List[Dict]) -> Tuple[List[str], str]:
    """LaMP-2: Movie tag classification.
    Profile has: description, tag, id
    """
    corpus = [f'{x.get("description", "")}' for x in profile]
    query = extract_after_keyword(inp, 'description:')
    return corpus, query


def lamp3_query_corpus(inp: str, profile: List[Dict]) -> Tuple[List[str], str]:
    """LaMP-3: Product rating prediction.
    Profile has: text, score, id
    """
    corpus = [f'{x.get("text", "")}' for x in profile]
    query = extract_after_keyword(inp, 'review:')
    return corpus, query


def lamp4_query_corpus(inp: str, profile: List[Dict]) -> Tuple[List[str], str]:
    """LaMP-4: News headline generation.
    Profile has: title, text, id
    """
    corpus = [f'{x.get("title", "")} {x.get("text", "")}' for x in profile]
    query = extract_after_keyword(inp, 'article:')
    return corpus, query


QUERY_CORPUS_MAKERS = {
    "lamp1": lamp1_query_corpus,
    "lamp2": lamp2_query_corpus,
    "lamp3": lamp3_query_corpus,
    "lamp4": lamp4_query_corpus,
}


# =============================================================================
# PROMPT GENERATORS FOR EACH TASK
# =============================================================================

def add_string_after_title(original_string: str, string_to_add: str) -> str:
    """Add context string after 'title' keyword."""
    title_index = original_string.find("title")
    if title_index == -1:
        return string_to_add + " " + original_string
    return original_string[:title_index+5] + ", and " + string_to_add + original_string[title_index+5:]


def create_lamp1_prompt(inp: str, profile: List[Dict], max_length: int, tokenizer) -> str:
    """LaMP-1: Citation classification prompt."""
    if not profile:
        return inp
    prompts = []
    per_p_max_length = max((max_length - 2 * (len(profile) - 1)) // len(profile), 10)
    saved_tokens = 0
    for p in profile:
        tokens = tokenizer(p["title"], max_length=per_p_max_length + saved_tokens - 2, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - 2
        new_title = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{new_title}"'
        prompts.append(prompt)
    return add_string_after_title(inp, ", and ".join(prompts))


def create_lamp2_prompt(inp: str, profile: List[Dict], max_length: int, tokenizer) -> str:
    """LaMP-2: Movie tag classification prompt.
    Profile has: description, tag, id
    PPEP format (Table 5): the tag for the movie: "[description]" is "[tag]"
    AIP format: concat(PPEP, ", and "). [INPUT]
    """
    if not profile:
        return inp
    per_p_max_length = max((max_length - 1 - 2 * (len(profile) - 1)) // len(profile), 10)
    saved_tokens = 0
    prompts = []
    for p in profile:
        tag = p.get("tag", "")
        needed_part_len = len(tokenizer(f'the tag for the movie: " " is "{tag}"')['input_ids'])
        desc = p.get("description", "")
        tokens = tokenizer(desc, max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'the tag for the movie: "{new_text}" is "{tag}"'
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. {inp}'


def create_lamp3_prompt(inp: str, profile: List[Dict], max_length: int, tokenizer) -> str:
    """LaMP-3: Product rating prediction prompt.
    Profile has: text, score, id
    PPEP format (Table 5): [score] is the score for "[text]"
    AIP format: concat(PPEP, ", and "). [INPUT]
    """
    if not profile:
        return inp
    per_p_max_length = max((max_length - 1 - 2 * (len(profile) - 1)) // len(profile), 10)
    saved_tokens = 0
    prompts = []
    for p in profile:
        score = p.get("score", "")
        needed_part_len = len(tokenizer(f'{score} is the score for " "')['input_ids'])
        text = p.get("text", "")
        tokens = tokenizer(text, max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'{score} is the score for "{new_text}"'
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. {inp}'


def create_lamp4_prompt(inp: str, profile: List[Dict], max_length: int, tokenizer) -> str:
    """LaMP-4: News headline generation prompt.
    Profile has: title, text, id
    PPEP format (Table 5): "[title]" is the title for "[text]"
    AIP format: concat(PPEP, ", and "). [INPUT]
    """
    if not profile:
        return inp
    per_p_max_length = max((max_length - 1 - 2 * (len(profile) - 1)) // len(profile), 10)
    saved_tokens = 0
    prompts = []
    for p in profile:
        title = p.get("title", "")
        needed_part_len = len(tokenizer(f'"{title}" is the title for " "')['input_ids'])
        text = p.get("text", "")
        tokens = tokenizer(text, max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{title}" is the title for "{new_text}"'
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. {inp}'


PROMPT_CREATORS = {
    "lamp1": create_lamp1_prompt,
    "lamp2": create_lamp2_prompt,
    "lamp3": create_lamp3_prompt,
    "lamp4": create_lamp4_prompt,
}


# =============================================================================
# LABELS FOR CLASSIFICATION TASKS
# =============================================================================

TASK_LABELS = {
    "lamp1": ["[1]", "[2]"],
    "lamp2": ['sci-fi', 'based on a book', 'comedy', 'action', 'twist ending', 'dystopia', 
               'dark comedy', 'classic', 'psychology', 'fantasy', 'romance', 'thought-provoking', 
               'social commentary', 'violence', 'true story'],
    "lamp3": ["1", "2", "3", "4", "5"],
    "lamp4": [],  # Generation task
}

# Task to dataset folder mapping
TASK_TO_DATASET = {
    "lamp1": "dataset1",
    "lamp2": "dataset2",
    "lamp3": "dataset3",
    "lamp4": "dataset4",
}


# =============================================================================
# DATASET HANDLING
# =============================================================================

def load_lamp_data(data_dir: str) -> List[Dict]:
    """
    Load LaMP data from directory structure:
    data_dir/
        inputs.json - list of {id, input, profile}
        outputs.json - list of {id, output}
    """
    inputs_path = os.path.join(data_dir, "inputs.json")
    outputs_path = os.path.join(data_dir, "outputs.json")
    
    with open(inputs_path, 'r', encoding='utf-8') as f:
        inputs_data = json.load(f)
    
    with open(outputs_path, 'r', encoding='utf-8') as f:
        outputs_data = json.load(f)
    
    # Handle different output formats
    if isinstance(outputs_data, dict) and 'golds' in outputs_data:
        outputs_data = outputs_data['golds']
    
    # Create output lookup
    output_lookup = {item['id']: item['output'] for item in outputs_data}
    
    # Merge inputs and outputs
    merged_data = []
    for item in inputs_data:
        item_id = item['id']
        if item_id in output_lookup:
            merged_data.append({
                'id': item_id,
                'input': item['input'],
                'profile': item.get('profile', []),
                'output': output_lookup[item_id]
            })
    
    return merged_data


class LaMPDataset(Dataset):
    """PyTorch Dataset for LaMP tasks with Contriever retrieval."""
    
    def __init__(
        self,
        data: List[Dict],
        task: str,
        tokenizer,
        retriever: Optional[ContrieverRetriever],
        num_retrieved: int = 4,
        max_length: int = 512,
        max_profile_size: int = 200,
    ):
        self.data = data
        self.task = task
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.num_retrieved = num_retrieved
        self.max_length = max_length
        self.max_profile_size = max_profile_size
        self.query_corpus_maker = QUERY_CORPUS_MAKERS[task]
        self.prompt_creator = PROMPT_CREATORS[task]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        inp = item['input']
        profile = item.get('profile', [])
        output = item['output']
        
        # Cap profile size to avoid slowdowns from huge profiles
        if len(profile) > self.max_profile_size:
            profile = profile[:self.max_profile_size]
        
        # Retrieve top-k profile items using Contriever
        if self.retriever and profile:
            corpus, query = self.query_corpus_maker(inp, profile)
            selected_profile = self.retriever.retrieve_top_k(
                corpus, profile, query, self.num_retrieved
            )
        else:
            selected_profile = profile[:self.num_retrieved] if profile else []
        
        # Generate prompt with profile context
        factor = 0.6
        while factor > 0:
            try:
                max_len_prompt = self.max_length - min(
                    len(self.tokenizer(inp)['input_ids']), 
                    int(factor * self.max_length)
                )
                source = self.prompt_creator(inp, selected_profile, max_len_prompt, self.tokenizer)
                break
            except:
                factor -= 0.1
                if factor <= 0:
                    source = inp
        
        return {
            "id": item['id'],
            "source": source,
            "target": output
        }


def convert_to_hf_dataset(dataset: LaMPDataset, cache_dir: str) -> datasets.Dataset:
    """Convert PyTorch Dataset to HuggingFace Dataset."""
    def gen():
        for idx in range(len(dataset)):
            yield dataset[idx]
    return datasets.Dataset.from_generator(gen, cache_dir=cache_dir)


def generate_and_evaluate(
    model, 
    tokenizer, 
    val_data: List[Dict], 
    task: str,
    retriever: Optional[ContrieverRetriever] = None,
    num_retrieved: int = 4,
    max_length: int = 512,
    max_new_tokens: int = 32, 
    max_profile_size: int = 200,
    device: str = "cuda:0",
    is_mamba2: bool = False,
    is_qwen3: bool = False,
) -> Dict:
    """
    Generate predictions and compute metrics for causal LM models.
    Uses profile-augmented prompts (same as training) for proper evaluation.
    """
    from tqdm import tqdm
    from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
    import math
    
    print(f"\nGenerating predictions for {len(val_data)} samples...")
    
    query_corpus_maker = QUERY_CORPUS_MAKERS.get(task)
    prompt_creator = PROMPT_CREATORS.get(task)
    
    predictions = []
    ground_truths = []
    
    model.eval()
    with torch.no_grad():
        for item in tqdm(val_data, desc="Generating"):
            inp = item['input']
            profile = item.get('profile', [])
            
            # Cap profile size to avoid slowdowns
            if len(profile) > max_profile_size:
                profile = profile[:max_profile_size]
            
            # Create profile-augmented prompt (same as training)
            if retriever and profile and query_corpus_maker and prompt_creator:
                corpus, query = query_corpus_maker(inp, profile)
                selected_profile = retriever.retrieve_top_k(corpus, profile, query, num_retrieved)
                
                # Generate prompt with profile context
                factor = 0.6
                while factor > 0:
                    try:
                        max_len_prompt = max_length - min(
                            len(tokenizer(inp)['input_ids']), 
                            int(factor * max_length)
                        )
                        augmented_input = prompt_creator(inp, selected_profile, max_len_prompt, tokenizer)
                        break
                    except:
                        factor -= 0.1
                        if factor <= 0:
                            augmented_input = inp
            else:
                augmented_input = inp
            
            # Format prompt based on model type
            if is_mamba2:
                # Mamba2: Simple prompt without instruction template
                prompt = f"{augmented_input}\nAnswer:"
            elif is_qwen3:
                # Qwen3: Use chat template format
                prompt = f"<|im_start|>user\n{augmented_input}<|im_end|>\n<|im_start|>assistant\n"
            else:
                # TinyLlama: Use instruction template
                prompt = f"<s>[INST] {augmented_input} [/INST]"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if is_mamba2:
                # Mamba2 uses different generate signature via mamba_ssm
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    max_new_tokens=max_new_tokens,
                )
            elif is_qwen3:
                # Qwen3: Use greedy decoding with proper EOS handling
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode only the generated part
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            predictions.append(prediction)
            ground_truths.append(item['output'].strip())
    
    # Compute task-appropriate metrics
    if task in ["lamp1", "lamp2"]:
        # Classification metrics: Accuracy and F1
        preds_lower = [p.lower() for p in predictions]
        golds_lower = [g.lower() for g in ground_truths]
        
        unique_labels = sorted(set(golds_lower))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        pred_indices = []
        gold_indices = []
        
        for pred, gold in zip(preds_lower, golds_lower):
            gold_idx = label_to_idx.get(gold, -1)
            pred_idx = label_to_idx.get(pred, -1)
            
            # Partial matching if no exact match
            if pred_idx == -1:
                for label, idx in label_to_idx.items():
                    if label in pred or pred in label:
                        pred_idx = idx
                        break
            
            pred_indices.append(pred_idx)
            gold_indices.append(gold_idx)
        
        correct = sum(1 for p, g in zip(pred_indices, gold_indices) if p == g and p != -1)
        total = len(predictions)
        valid = sum(1 for p in pred_indices if p != -1)
        accuracy = correct / total if total > 0 else 0.0
        
        valid_pairs = [(p, g) for p, g in zip(pred_indices, gold_indices) if p != -1 and g != -1]
        if valid_pairs:
            valid_preds, valid_golds = zip(*valid_pairs)
            present_labels = sorted(set(valid_golds))
            try:
                f1 = f1_score(valid_golds, valid_preds, labels=present_labels, average='macro')
            except:
                f1 = 0.0
        else:
            f1 = 0.0
        
        results = {"accuracy": accuracy, "f1": f1, "correct": correct, "total": total, "valid": valid}
        
    elif task == "lamp3":
        # Regression metrics: RMSE and MAE
        import re
        pred_scores = []
        gold_scores = []
        invalid_count = 0
        
        for pred, gold in zip(predictions, ground_truths):
            try:
                gold_score = float(gold)
            except ValueError:
                continue
            
            try:
                pred_score = float(pred)
            except ValueError:
                numbers = re.findall(r'[-+]?\d*\.?\d+', pred)
                if numbers:
                    pred_score = float(numbers[0])
                else:
                    invalid_count += 1
                    continue
            
            pred_scores.append(pred_score)
            gold_scores.append(gold_score)
        
        if pred_scores:
            mse = mean_squared_error(gold_scores, pred_scores)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(gold_scores, pred_scores)
        else:
            rmse = float('inf')
            mae = float('inf')
        
        results = {"rmse": rmse, "mae": mae, "total": len(predictions), 
                   "valid": len(pred_scores), "invalid": invalid_count}
        
    elif task == "lamp4":
        # Text generation metrics: ROUGE-1 and ROUGE-L
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rougeL_scores = []
            
            for pred, gold in zip(predictions, ground_truths):
                if not pred:
                    rouge1_scores.append(0.0)
                    rougeL_scores.append(0.0)
                    continue
                
                scores = scorer.score(gold, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            import numpy as np
            results = {
                "rouge1": float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
                "rougeL": float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
                "total": len(predictions)
            }
        except ImportError:
            print("Warning: rouge_score not installed. Install with: pip install rouge-score")
            results = {"rouge1": 0.0, "rougeL": 0.0, "total": len(predictions), "error": "rouge_score not installed"}
    else:
        # Default to classification
        correct = sum(1 for p, g in zip(predictions, ground_truths) if p.lower() == g.lower())
        results = {"accuracy": correct / len(predictions) if predictions else 0.0, "total": len(predictions)}
    
    # Show sample predictions
    print("\nSample predictions:")
    for i in range(min(10, len(predictions))):
        pred, gold = predictions[i], ground_truths[i]
        match = "✓" if pred.lower() == gold.lower() else "✗"
        print(f"  {match} Pred: '{pred[:50]}' | Gold: '{gold[:50]}'")
    
    # Save predictions
    pred_data = [{"prediction": p, "ground_truth": g} for p, g in zip(predictions, ground_truths)]
    
    return results, pred_data


# =============================================================================
# METRICS
# =============================================================================

def postprocess_text_classification(preds: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Postprocess text for classification metrics."""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


def postprocess_text_generation(preds: List[str], labels: List[str]) -> Tuple[List[str], List[List[str]]]:
    """Postprocess text for generation metrics."""
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def create_metric_f1_accuracy(tokenizer, all_labels: List[str]):
    """Create F1/Accuracy metric for classification tasks (LaMP-1, LaMP-2)."""
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    
    def create_mapping(x):
        x = x.strip()
        try:
            return all_labels.index(x)
        except:
            # Try lowercase matching
            x_lower = x.lower()
            for i, label in enumerate(all_labels):
                if label.lower() == x_lower:
                    return i
            return -1
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        
        pred_indices = [create_mapping(x) for x in decoded_preds]
        label_indices = [create_mapping(x) for x in decoded_labels]
        
        # Filter out invalid mappings for accuracy
        valid_pairs = [(p, l) for p, l in zip(pred_indices, label_indices) if p != -1 and l != -1]
        
        if valid_pairs:
            valid_preds, valid_labels = zip(*valid_pairs)
            accuracy = sum(1 for p, l in valid_pairs if p == l) / len(valid_pairs)
            
            # Compute F1 only on present labels
            present_labels = sorted(set(valid_labels))
            try:
                f1 = f1_metric.compute(
                    predictions=valid_preds, 
                    references=valid_labels, 
                    labels=present_labels,
                    average="macro"
                )["f1"]
            except:
                f1 = 0.0
        else:
            accuracy = 0.0
            f1 = 0.0
        
        return {"accuracy": accuracy, "f1": f1}
    
    return compute_metrics


def create_metric_mae_rmse(tokenizer, all_labels: List[str]):
    """Create MAE/RMSE metric for LaMP-3 (score prediction)."""
    mse_metric = evaluate.load("mse")
    mae_metric = evaluate.load("mae")
    
    def create_mapping(x, y):
        try:
            return float(x)
        except:
            y = float(y)
            return 1.0 if abs(1 - y) > abs(5 - y) else 5.0
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x, x) for x in decoded_labels]
        result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared=False)
        return {"mae": result_mae["mae"], "rmse": result_rmse["mse"]}
    
    return compute_metrics


def create_metric_bleu_rouge_meteor(tokenizer):
    """Create BLEU/ROUGE/METEOR metrics for generation tasks (LaMP-4)."""
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)
        result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {
            "bleu": result_bleu["score"],
            "rouge-1": result_rouge["rouge1"],
            "rouge-2": result_rouge["rouge2"],
            "rouge-L": result_rouge["rougeL"],
            "meteor": result_meteor["meteor"]
        }
    
    return compute_metrics


# =============================================================================
# PREPROCESSORS
# =============================================================================

def create_seq2seq_preprocessor(tokenizer, max_length: int):
    """Create preprocessor for Seq2Seq models (Flan-T5)."""
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        return model_inputs
    return preprocess_dataset


def create_causal_preprocessor(tokenizer, max_length: int, is_mamba2: bool = False, is_qwen3: bool = False):
    """Create preprocessor for Causal LM models (TinyLlama, Mamba2, or Qwen3)."""
    def preprocess_dataset(examples):
        texts = []
        for source, target in zip(examples["source"], examples["target"]):
            if is_mamba2:
                # Mamba2: Simple format without instruction template
                text = f"{source}\nAnswer: {target}"
            elif is_qwen3:
                # Qwen3: Use chat template format
                # Qwen3 supports /no_think for non-thinking mode
                text = f"<|im_start|>user\n{source}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>"
            else:
                # TinyLlama: Format with instruction template
                text = f"<s>[INST] {source} [/INST] {target}</s>"
            texts.append(text)
        
        model_inputs = tokenizer(
            texts, 
            max_length=max_length, 
            truncation=True,
            padding=False,
        )
        # Create labels as a proper list copy (not reference)
        model_inputs["labels"] = [ids.copy() if hasattr(ids, 'copy') else list(ids) for ids in model_inputs["input_ids"]]
        return model_inputs
    return preprocess_dataset


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_single_task(opts, task: str, data_dir: str, output_dir: str, retriever=None) -> Dict:
    """Train a single LaMP task and return results."""
    print(f"\n{'='*60}")
    print(f"Training {task}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print(f"Loading model: {opts.model_name}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_qwen3 = opts.model_type == "qwen3"
    
    if opts.model_type == "mamba2":
        # Mamba2 uses EleutherAI/gpt-neox-20b tokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", cache_dir=opts.cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # CRITICAL: Left padding is REQUIRED for Mamba2!
        tokenizer.padding_side = "left"
        print(f"   Using tokenizer: EleutherAI/gpt-neox-20b (padding: {tokenizer.padding_side})")
        
        # Load model using mamba_ssm package (NOT HuggingFace transformers!)
        print("   Loading Mamba2 backbone via mamba_ssm...")
        mamba_backbone = MambaLMHeadModel.from_pretrained(
            opts.model_name,
            device=device,
            dtype=torch.bfloat16 if opts.bf16 else (torch.float16 if opts.fp16 else torch.float32),
        )
        
        # Get config from the loaded model
        mamba_config = mamba_backbone.config
        print(f"   Config: d_model={mamba_config.d_model}, n_layer={mamba_config.n_layer}")
        
        # Create HF-compatible config wrapper
        config = Mamba2ConfigWrapper(mamba_config)
        
        # Create our causal LM wrapper
        model = Mamba2ForCausalLM(config, backbone=mamba_backbone)
        is_seq2seq = False
        is_mamba2 = True
    elif opts.model_type == "qwen3":
        # Qwen3-1.7B model loading
        tokenizer = AutoTokenizer.from_pretrained(
            opts.model_name, 
            cache_dir=opts.cache_dir,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # Left padding for generation
        tokenizer.padding_side = "left"
        print(f"   Using Qwen3 tokenizer (padding: {tokenizer.padding_side})")
        
        model = AutoModelForCausalLM.from_pretrained(
            opts.model_name,
            cache_dir=opts.cache_dir,
            torch_dtype=torch.bfloat16 if opts.bf16 else (torch.float16 if opts.fp16 else torch.float32),
            trust_remote_code=True,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        is_seq2seq = False
        is_mamba2 = False
    else:
        tokenizer = AutoTokenizer.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
        is_mamba2 = False
        
        if opts.model_type == "flan-t5":
            model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
            is_seq2seq = True
        else:  # tinyllama
            model = AutoModelForCausalLM.from_pretrained(
                opts.model_name, 
                cache_dir=opts.cache_dir,
                torch_dtype=torch.float16 if opts.fp16 else (torch.bfloat16 if opts.bf16 else torch.float32),
            )
            is_seq2seq = False
            # Set pad token for causal LM
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.eos_token_id
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    if is_mamba2:
        # Mamba2: Target in_proj and x_proj ONLY (NOT out_proj/conv1d per PEFT#2556)
        lora_config = LoraConfig(
            r=opts.lora_r,
            lora_alpha=opts.lora_alpha,
            lora_dropout=opts.lora_dropout,
            target_modules=["in_proj", "x_proj"],
            bias="none",
        )
    elif is_seq2seq:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=opts.lora_r,
            lora_alpha=opts.lora_alpha,
            lora_dropout=opts.lora_dropout,
            target_modules=["q", "v"],
        )
    elif is_qwen3:
        # Qwen3: Target attention projection layers
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=opts.lora_r,
            lora_alpha=opts.lora_alpha,
            lora_dropout=opts.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    else:  # tinyllama
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=opts.lora_r,
            lora_alpha=opts.lora_alpha,
            lora_dropout=opts.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    print("Loading training data...")
    train_data = load_lamp_data(os.path.join(data_dir, "train"))
    print(f"Loaded {len(train_data)} training samples")
    
    print("Loading validation data...")
    val_data = load_lamp_data(os.path.join(data_dir, "validation"))
    print(f"Loaded {len(val_data)} validation samples")
    
    # Debug mode: limit samples and epochs for quick testing
    if opts.debug:
        debug_train_samples = 8
        debug_val_samples = 4
        debug_epochs = 1
        print(f"\n*** DEBUG MODE ENABLED ***")
        print(f"  Limiting training samples: {len(train_data)} -> {min(debug_train_samples, len(train_data))}")
        print(f"  Limiting validation samples: {len(val_data)} -> {min(debug_val_samples, len(val_data))}")
        print(f"  Limiting epochs: {opts.num_epochs} -> {debug_epochs}")
        train_data = train_data[:debug_train_samples]
        val_data = val_data[:debug_val_samples]
        opts.num_epochs = debug_epochs
        print(f"*** DEBUG MODE: Quick test run ***\n")
    
    # Create PyTorch datasets
    train_dataset = LaMPDataset(
        train_data, task, tokenizer, retriever, 
        opts.num_retrieved, opts.max_length, opts.max_profile_size
    )
    val_dataset = LaMPDataset(
        val_data, task, tokenizer, retriever,
        opts.num_retrieved, opts.max_length, opts.max_profile_size
    )
    
    # Convert to HuggingFace datasets
    print("Converting to HuggingFace datasets...")
    train_hf = convert_to_hf_dataset(train_dataset, cache_dir=opts.cache_dir)
    val_hf = convert_to_hf_dataset(val_dataset, cache_dir=opts.cache_dir)
    
    # Move retriever to CPU temporarily to free GPU memory for training
    if retriever:
        retriever.to_cpu()
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    if is_seq2seq:
        preprocessor = create_seq2seq_preprocessor(tokenizer, opts.max_length)
    else:
        preprocessor = create_causal_preprocessor(tokenizer, opts.max_length, is_mamba2=is_mamba2, is_qwen3=is_qwen3)
    
    train_hf = train_hf.map(preprocessor, batched=True, remove_columns=train_hf.column_names)
    val_hf = val_hf.map(preprocessor, batched=True, remove_columns=val_hf.column_names)
    
    # Setup metrics
    labels = TASK_LABELS[task]
    if task in ["lamp1", "lamp2"]:
        compute_metrics = create_metric_f1_accuracy(tokenizer, labels)
        best_metric = "accuracy"
        greater_is_better = True
    elif task == "lamp3":
        compute_metrics = create_metric_mae_rmse(tokenizer, labels)
        best_metric = "mae"
        greater_is_better = False
    else:  # LaMP-4
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer)
        best_metric = "rouge-1"
        greater_is_better = True
    
    # Setup data collator - use DataCollatorForSeq2Seq for both model types
    # It properly pads both input_ids and labels
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model, 
        padding=True,
        max_length=opts.max_length,
        label_pad_token_id=-100
    )
    
    # Setup NVML callback for power/VRAM monitoring during training steps
    nvml_callback = TrainingStepNVMLCallback(gpu_index=0, output_dir=output_dir, sample_every_n_steps=1)
    
    # Setup training arguments
    if is_seq2seq:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=opts.batch_size,
            per_device_eval_batch_size=opts.batch_size,
            gradient_accumulation_steps=opts.gradient_accumulation_steps,
            learning_rate=opts.learning_rate,
            weight_decay=opts.weight_decay,
            num_train_epochs=opts.num_epochs,
            lr_scheduler_type="linear",
            warmup_ratio=opts.warmup_ratio,
            generation_num_beams=opts.generation_num_beams,
            predict_with_generate=True,
            save_strategy="no",  # Only save final checkpoint manually
            logging_steps=50,
            eval_accumulation_steps=1,
            generation_max_length=opts.generation_max_length,
            fp16=opts.fp16,
            bf16=opts.bf16,
            seed=opts.seed,
            report_to="none",
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[nvml_callback],
        )
    else:
        # For causal LM, we use loss as metric since regular Trainer doesn't do generation
        training_args = TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=opts.batch_size,
            per_device_eval_batch_size=opts.batch_size,
            gradient_accumulation_steps=opts.gradient_accumulation_steps,
            learning_rate=opts.learning_rate,
            weight_decay=opts.weight_decay,
            num_train_epochs=opts.num_epochs,
            lr_scheduler_type="linear",
            warmup_ratio=opts.warmup_ratio,
            save_strategy="no",  # Only save final checkpoint manually
            logging_steps=50,
            fp16=opts.fp16,
            bf16=opts.bf16,
            seed=opts.seed,
            report_to="none",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            tokenizer=tokenizer,
            callbacks=[nvml_callback],
        )
    
    # Train
    print("Starting training...")
    train_start_time = time.time()
    trainer.train()
    train_time_secs = time.time() - train_start_time
    print(f"Training completed in {train_time_secs/60:.1f} minutes")
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    
    if is_seq2seq:
        # For Seq2Seq models, use Trainer's built-in evaluation with generation
        results = trainer.evaluate()
    else:
        # For Causal LM, use our custom generation-based evaluation
        print("Running generation-based evaluation for Causal LM...")
        # Move retriever back to GPU for evaluation
        if retriever:
            retriever.to_gpu(str(next(model.parameters()).device))
        device = str(next(model.parameters()).device)
        gen_results, pred_data = generate_and_evaluate(
            model, tokenizer, val_data, task,
            retriever=retriever,
            num_retrieved=opts.num_retrieved,
            max_length=opts.max_length,
            max_new_tokens=32, 
            max_profile_size=opts.max_profile_size,
            device=device,
            is_mamba2=is_mamba2,
            is_qwen3=is_qwen3,
        )
        # Combine with trainer's loss-based results
        loss_results = trainer.evaluate()
        results = {**loss_results, **gen_results}
        
        # Save predictions
        preds_path = os.path.join(output_dir, "predictions.json")
        with open(preds_path, 'w') as f:
            json.dump(pred_data, f, indent=2)
        print(f"Predictions saved to {preds_path}")
    
    print(f"\nValidation Results for {task}:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Aggregate NVML stats from training step callback
    avg_power_w = nvml_callback.get_avg_power()
    avg_vram_used_mb = nvml_callback.get_avg_vram_mb()
    energy_Wh = nvml_callback.get_energy_wh()
    
    # Get detailed NVML summary
    nvml_summary = nvml_callback.summary if hasattr(nvml_callback, "summary") else {}
    max_power_w = nvml_summary.get("max_power_watts")
    max_vram_mb = nvml_summary.get("max_vram_used_mb")
    num_step_samples = nvml_summary.get("num_step_samples", 0)
    compute_ratio = nvml_summary.get("compute_ratio")
    
    # Get primary accuracy metric for SAM calculation
    primary_accuracy = None
    if task in ["lamp1", "lamp2"]:
        primary_accuracy = results.get("accuracy") or results.get("eval_accuracy")
    elif task == "lamp3":
        # For regression tasks, use inverse normalized MAE as "accuracy" proxy
        mae = results.get("mae") or results.get("eval_mae") or results.get("rmse") or results.get("eval_rmse")
        if mae is not None and mae > 0:
            # Normalize: assume score range 1-5, so max MAE is 4
            primary_accuracy = max(0, 1 - mae / 4.0)
    elif task == "lamp4":
        # For generation tasks, use ROUGE-1 as accuracy proxy
        primary_accuracy = results.get("rouge1") or results.get("rouge-1") or results.get("eval_rouge-1")
    
    # Calculate SAM metrics
    sam_metrics = {}
    if primary_accuracy is not None and energy_Wh is not None and energy_Wh > 0:
        sam_metrics = calculate_all_sam_metrics(primary_accuracy, energy_Wh)
        print(f"\n--- SAM Metrics (Sustainable AI Metric) ---")
        print(f"  Primary accuracy used: {primary_accuracy:.4f}")
        print(f"  Energy consumption: {energy_Wh:.4f} Wh")
        for sam_key, sam_val in sam_metrics.items():
            if sam_val is not None:
                print(f"  {sam_key}: {sam_val:.6f}")
        print(f"-------------------------------------------")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    
    # Save comprehensive benchmark results
    benchmark_results = {
        "task": task,
        "method": "LoRA",
        "model_type": opts.model_type,
        "model_name": opts.model_name,
        "batch_size": opts.batch_size,
        "learning_rate": opts.learning_rate,
        "epochs": opts.num_epochs,
        "max_length": opts.max_length,
        "lora_r": opts.lora_r,
        "lora_alpha": opts.lora_alpha,
        "lora_dropout": opts.lora_dropout,
        "num_retrieved": opts.num_retrieved,
        "validation_results": results,
        
        # Training time metrics
        "training_time_seconds": train_time_secs,
        "training_time_minutes": train_time_secs / 60.0,
        
        # Parameter counts
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
        "trainable_percentage": float(trainable_pct),
        
        # GPU power metrics (sampled during training steps)
        "avg_gpu_power_watts": avg_power_w,
        "max_gpu_power_watts": max_power_w,
        
        # GPU VRAM metrics (sampled during training steps)
        "avg_gpu_vram_used_mb": avg_vram_used_mb,
        "max_gpu_vram_used_mb": max_vram_mb,
        
        # Energy consumption
        "energy_consumption_wh": energy_Wh,
        
        # SAM (Sustainable AI Metric) values
        "sam_metrics": sam_metrics,
        "primary_accuracy_for_sam": primary_accuracy,
        
        # Sampling info
        "num_nvml_step_samples": num_step_samples,
        "compute_ratio": compute_ratio,
    }
    
    benchmark_path = os.path.join(output_dir, "benchmark_results.json")
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"Benchmark results saved to {benchmark_path}")
    
    # Also save validation results separately
    results_path = os.path.join(output_dir, "validation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Validation results saved to {results_path}")
    
    # Print benchmark summary
    print(f"\n{'='*60}")
    print(f"TRAINING BENCHMARK SUMMARY - {task}")
    print(f"{'='*60}")
    print(f"  Model: {opts.model_name}")
    print(f"  Training time: {train_time_secs:.1f} seconds ({train_time_secs/60:.1f} minutes)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    if avg_power_w is not None:
        print(f"  Avg GPU power (during training): {avg_power_w:.2f} W")
    if max_power_w is not None:
        print(f"  Max GPU power (during training): {max_power_w:.2f} W")
    if avg_vram_used_mb is not None:
        print(f"  Avg VRAM used (during training): {avg_vram_used_mb:.0f} MiB")
    if max_vram_mb is not None:
        print(f"  Max VRAM used (during training): {max_vram_mb:.0f} MiB")
    if energy_Wh is not None:
        print(f"  Energy consumption: {energy_Wh:.4f} Wh")
    if sam_metrics:
        print(f"  SAM_1: {sam_metrics.get('SAM_1', 'N/A')}")
        print(f"  SAM_2: {sam_metrics.get('SAM_2', 'N/A')}")
        print(f"  SAM_5: {sam_metrics.get('SAM_5', 'N/A')}")
    if num_step_samples > 0:
        print(f"  NVML samples collected: {num_step_samples}")
    if compute_ratio is not None:
        print(f"  Compute ratio: {compute_ratio:.2%}")
    print(f"{'='*60}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Clean up GPU memory
    del model
    del trainer
    torch.cuda.empty_cache()
    
    return results


def evaluate_only(opts, task: str, data_dir: str, lora_path: str, retriever=None) -> Dict:
    """Evaluate a pretrained LoRA model without training."""
    print(f"\n{'='*60}")
    print(f"EVAL-ONLY MODE: {task}")
    print(f"Data: {data_dir}")
    print(f"LoRA Path: {lora_path}")
    print(f"{'='*60}\n")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_qwen3 = opts.model_type == "qwen3"
    
    # Load tokenizer and model
    print(f"Loading base model: {opts.model_name}")
    
    if opts.model_type == "mamba2":
        # Mamba2 uses EleutherAI/gpt-neox-20b tokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", cache_dir=opts.cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # CRITICAL: Left padding is REQUIRED for Mamba2!
        tokenizer.padding_side = "left"
        
        # Load model using mamba_ssm package
        print("   Loading Mamba2 backbone via mamba_ssm...")
        mamba_backbone = MambaLMHeadModel.from_pretrained(
            opts.model_name,
            device=device,
            dtype=torch.bfloat16 if opts.bf16 else torch.float16,
        )
        
        # Get config from the loaded model
        mamba_config = mamba_backbone.config
        config = Mamba2ConfigWrapper(mamba_config)
        
        # Create our causal LM wrapper
        model = Mamba2ForCausalLM(config, backbone=mamba_backbone)
        is_seq2seq = False
        is_mamba2 = True
    elif opts.model_type == "qwen3":
        # Qwen3 model loading
        tokenizer = AutoTokenizer.from_pretrained(
            opts.model_name,
            cache_dir=opts.cache_dir,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        model = AutoModelForCausalLM.from_pretrained(
            opts.model_name,
            cache_dir=opts.cache_dir,
            torch_dtype=torch.bfloat16 if opts.bf16 else torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        is_seq2seq = False
        is_mamba2 = False
    else:
        tokenizer = AutoTokenizer.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
        is_mamba2 = False
        
        if opts.model_type == "flan-t5":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                opts.model_name, 
                cache_dir=opts.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            is_seq2seq = True
        else:  # tinyllama
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                opts.model_name, 
                cache_dir=opts.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            is_seq2seq = False
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from {lora_path}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    
    # Load validation data
    print("Loading validation data...")
    val_data = load_lamp_data(os.path.join(data_dir, "validation"))
    print(f"Loaded {len(val_data)} validation samples")
    
    if opts.max_samples:
        val_data = val_data[:opts.max_samples]
        print(f"Limiting to {opts.max_samples} samples")
    
    # Run evaluation
    print("\nRunning evaluation...")
    device = str(next(model.parameters()).device)
    
    if is_seq2seq:
        # For Seq2Seq, use generation-based evaluation similar to causal
        from tqdm import tqdm
        predictions = []
        ground_truths = []
        
        query_corpus_maker = QUERY_CORPUS_MAKERS.get(task)
        prompt_creator = PROMPT_CREATORS.get(task)
        
        model.eval()
        with torch.no_grad():
            for item in tqdm(val_data, desc="Generating"):
                inp = item['input']
                profile = item.get('profile', [])
                
                if len(profile) > opts.max_profile_size:
                    profile = profile[:opts.max_profile_size]
                
                # Create profile-augmented prompt
                if retriever and profile and query_corpus_maker and prompt_creator:
                    corpus, query = query_corpus_maker(inp, profile)
                    selected_profile = retriever.retrieve_top_k(corpus, profile, query, opts.num_retrieved)
                    
                    factor = 0.6
                    while factor > 0:
                        try:
                            max_len_prompt = opts.max_length - min(
                                len(tokenizer(inp)['input_ids']), 
                                int(factor * opts.max_length)
                            )
                            augmented_input = prompt_creator(inp, selected_profile, max_len_prompt, tokenizer)
                            break
                        except:
                            factor -= 0.1
                            if factor <= 0:
                                augmented_input = inp
                else:
                    augmented_input = inp
                
                inputs = tokenizer(augmented_input, return_tensors="pt", truncation=True, max_length=opts.max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    num_beams=4,
                )
                
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                predictions.append(prediction)
                ground_truths.append(item['output'].strip())
        
        # Simple accuracy for now
        correct = sum(1 for p, g in zip(predictions, ground_truths) if p.lower() == g.lower())
        results = {"accuracy": correct / len(predictions), "total": len(predictions)}
        pred_data = [{"prediction": p, "ground_truth": g} for p, g in zip(predictions, ground_truths)]
    else:
        # For Causal LM, use the enhanced generate_and_evaluate
        gen_results, pred_data = generate_and_evaluate(
            model, tokenizer, val_data, task,
            retriever=retriever,
            num_retrieved=opts.num_retrieved,
            max_length=opts.max_length,
            max_new_tokens=32, 
            max_profile_size=opts.max_profile_size,
            device=device,
            is_mamba2=is_mamba2,
            is_qwen3=is_qwen3,
        )
        results = gen_results
    
    print(f"\nEvaluation Results for {task}:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    output_dir = os.path.dirname(lora_path) if os.path.isdir(lora_path) else lora_path
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save predictions
    preds_path = os.path.join(output_dir, "predictions.json")
    with open(preds_path, 'w') as f:
        json.dump(pred_data, f, indent=2)
    print(f"Predictions saved to {preds_path}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LoRA Finetuning for LaMP Tasks")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, required=True, choices=["flan-t5", "tinyllama", "mamba2", "qwen3"],
                        help="Model type: 'flan-t5', 'tinyllama', 'mamba2', or 'qwen3'")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Specific model name (default: auto-select based on model_type)")
    
    # Task/Data arguments
    parser.add_argument("--task", type=str, required=True,
                        help="LaMP task(s): lamp1, lamp2, lamp3, lamp4, 'all', or comma-separated like 'lamp1,lamp3,lamp4'")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory (for single task) or base directory containing dataset1-4 folders (for --task all)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model checkpoints")
    
    # Retrieval arguments
    parser.add_argument("--num_retrieved", type=int, default=4,
                        help="Number of profile items to retrieve (default: 4)")
    parser.add_argument("--max_profile_size", type=int, default=100,
                        help="Max profile items to consider for retrieval (default: 100). Caps large profiles to speed up processing.")
    parser.add_argument("--use_retrieval", action="store_true", default=True,
                        help="Use Contriever retrieval (default: True)")
    parser.add_argument("--no_retrieval", action="store_false", dest="use_retrieval",
                        help="Disable Contriever retrieval")
    
    # Training arguments
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--generation_max_length", type=int, default=128,
                        help="Maximum generation length (default: 128)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per device (default: 4)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio (default: 0.05)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--generation_num_beams", type=int, default=4,
                        help="Number of beams for generation (default: 4)")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout (default: 0.1)")
    
    # Other arguments
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory (default: ./cache)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 training")
    
    # Eval-only mode arguments
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and only run evaluation on a pretrained model")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to pretrained LoRA adapter for eval-only mode")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of evaluation samples (for quick testing)")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: use only 8 train samples, 4 val samples, 1 epoch for quick testing")
    
    opts = parser.parse_args()
    
    # Set default model names
    if opts.model_name is None:
        if opts.model_type == "flan-t5":
            opts.model_name = "google/flan-t5-base"
        elif opts.model_type == "mamba2":
            opts.model_name = "state-spaces/mamba2-1.3b"
        elif opts.model_type == "qwen3":
            opts.model_name = "Qwen/Qwen3-1.7B"
        else:  # tinyllama
            opts.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Check mamba_ssm availability for mamba2
    if opts.model_type == "mamba2" and not MAMBA_SSM_OK:
        raise ImportError(
            "mamba_ssm package is required for mamba2 models.\n"
            "Install with: pip install mamba-ssm\n"
            "Also requires: pip install causal-conv1d>=1.2.0"
        )
    
    print(f"{'='*60}")
    mode_str = "Eval-Only" if opts.eval_only else "LoRA Finetuning"
    print(f"{mode_str} for LaMP Tasks")
    if opts.debug:
        print(f"*** DEBUG MODE ENABLED (8 train, 4 val, 1 epoch) ***")
    print(f"{'='*60}")
    print(f"Model: {opts.model_name}")
    print(f"Task(s): {opts.task}")
    print(f"Data directory: {opts.data_dir}")
    if opts.eval_only:
        print(f"LoRA path: {opts.lora_path or 'auto-detect'}")
    else:
        print(f"Output directory: {opts.output_dir}")
    print(f"Num retrieved: {opts.num_retrieved}")
    if opts.debug:
        print(f"Debug mode: ON")
    
    # Create cache directory
    os.makedirs(opts.cache_dir, exist_ok=True)
    os.makedirs(opts.output_dir, exist_ok=True)
    
    # Determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Initialize Contriever retriever once (shared across tasks)
    retriever = None
    if opts.use_retrieval:
        print("Loading Contriever retriever...")
        retriever = ContrieverRetriever(device=device)
    
    all_results = {}
    
    # Parse task argument - can be single task, 'all', or comma-separated
    if opts.task == "all":
        tasks = ["lamp1", "lamp2", "lamp3", "lamp4"]
    else:
        tasks = [t.strip().lower() for t in opts.task.split(",")]
        # Validate tasks
        valid_tasks = {"lamp1", "lamp2", "lamp3", "lamp4"}
        invalid = set(tasks) - valid_tasks
        if invalid:
            print(f"ERROR: Invalid task(s): {invalid}. Valid tasks: {valid_tasks}")
            return
    
    print(f"\nRunning {len(tasks)} task(s): {', '.join(tasks)}")
    
    for task in tasks:
        if len(tasks) > 1:
            # Multiple tasks - expect data in data_dir/dataset{N}
            dataset_folder = TASK_TO_DATASET[task]
            task_data_dir = os.path.join(opts.data_dir, dataset_folder)
            task_output_dir = os.path.join(opts.output_dir, task)
        else:
            # Single task - use data_dir directly
            task_data_dir = opts.data_dir
            task_output_dir = opts.output_dir
        
        # Check if data directory exists
        if not os.path.exists(task_data_dir):
            print(f"\nWARNING: Data directory not found for {task}: {task_data_dir}")
            print(f"Skipping {task}...")
            all_results[task] = {"error": f"Data directory not found: {task_data_dir}"}
            continue
        
        try:
            # Move retriever back to GPU if needed
            if retriever:
                retriever.to_gpu(device)
            
            if opts.eval_only:
                # Eval-only mode: use provided lora_path or construct from output_dir
                lora_path = opts.lora_path
                if lora_path is None:
                    # Try to find the final_model in task output dir
                    lora_path = os.path.join(task_output_dir, "final_model")
                    if not os.path.exists(lora_path):
                        lora_path = task_output_dir
                
                if not os.path.exists(lora_path):
                    print(f"ERROR: LoRA path not found: {lora_path}")
                    print("Please specify --lora_path for eval-only mode")
                    all_results[task] = {"error": f"LoRA path not found: {lora_path}"}
                    continue
                
                results = evaluate_only(opts, task, task_data_dir, lora_path, retriever)
            else:
                # Training mode
                results = train_single_task(opts, task, task_data_dir, task_output_dir, retriever)
            
            all_results[task] = results
        except Exception as e:
            print(f"ERROR {'evaluating' if opts.eval_only else 'training'} {task}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task] = {"error": str(e)}
    
    # Save combined results if multiple tasks
    if len(tasks) > 1:
        combined_results_path = os.path.join(opts.output_dir, "all_tasks_results.json")
        with open(combined_results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nCombined results saved to {combined_results_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*60}")
    for task, results in all_results.items():
        if "error" in results:
            print(f"{task}: ERROR - {results['error']}")
        else:
            # Extract key metrics for display
            key_metrics = []
            for k, v in results.items():
                if isinstance(v, (int, float)) and not k.startswith("eval_runtime"):
                    if "eval_" in k:
                        clean_k = k.replace("eval_", "")
                    else:
                        clean_k = k
                    key_metrics.append(f"{clean_k}: {v:.4f}")
            print(f"{task}: {', '.join(key_metrics[:4])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

