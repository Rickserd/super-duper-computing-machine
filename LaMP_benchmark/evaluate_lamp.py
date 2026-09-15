"""
Evaluation script for fine-tuned (and zero-shot) LaMP models.
Generates predictions and computes accuracy/F1 metrics.
Supports Flan-T5 (seq2seq), TinyLlama (causal LM), Mamba2, Mamba-1 1.4B, and Qwen3-1.7B models.

Uses the same profile-augmented prompt construction as training.

Usage (fine-tuned model with LoRA adapter):
    # For TinyLlama (causal LM)
    python evaluate_lamp.py --model_path ./test/final_model --data_dir ./dataset2 --task lamp2 --model_type tinyllama

    # For Flan-T5 (seq2seq)
    python evaluate_lamp.py --model_path ./test/final_model --data_dir ./dataset2 --task lamp2 --model_type flan-t5

    # For Mamba2
    python evaluate_lamp.py --model_path ./test/final_model --data_dir ./dataset2 --task lamp2 --model_type mamba2

    # For Mamba-1 1.4B (HuggingFace-compatible, no mamba_ssm needed)
    python evaluate_lamp.py --model_path ./test/final_model --data_dir ./dataset2 --task lamp2 --model_type mamba1

    # For Qwen3-1.7B
    python evaluate_lamp.py --model_path ./test/final_model --data_dir ./dataset2 --task lamp2 --model_type qwen3

Usage (zero-shot evaluation — no fine-tuning, base model only):
    python evaluate_lamp.py --zero_shot --data_dir ./dataset2 --task lamp2 --model_type tinyllama --output_dir ./zero_shot_results

    python evaluate_lamp.py --zero_shot --data_dir ./dataset1 --task lamp1 --model_type qwen3 --output_dir ./zero_shot_results

Usage (full fine-tuning checkpoint — entire model saved, no adapter):
    python evaluate_lamp.py --model_path ./full_ft_output/final_model --data_dir ./dataset2 --task lamp2 --model_type tinyllama --method full_ft

Usage (BitFit checkpoint — entire model saved, no adapter):
    python evaluate_lamp.py --model_path ./bitfit_output/final_model --data_dir ./dataset2 --task lamp2 --model_type tinyllama --method bitfit

Usage (QLoRA — quantized base model + LoRA adapter):
    python evaluate_lamp.py --model_path ./qlora_output/final_model --data_dir ./dataset2 --task lamp2 --model_type tinyllama --method qlora --bf16
    python evaluate_lamp.py --model_path ./qlora_output/final_model --data_dir ./dataset1 --task lamp1 --model_type qwen3 --method qlora --qlora_bits 4 --bf16
"""

import argparse
import json
import os
import time
import threading
import subprocess
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from peft import PeftModel
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
import numpy as np
import math

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not installed. Install with: pip install rouge-score")


# =============================================================================
# GPU INFERENCE TRACKER (time, power, energy, VRAM)
# =============================================================================

class GPUInferenceTracker:
    """Tracks per-sample and aggregate inference metrics: time, power, energy, VRAM.

    Separates **total time** (retrieval + tokenization + generation + decoding)
    from **generation time** (model.generate only).  Also records output token
    counts so it can compute throughput and inference latency.

    Uses pynvml when available; falls back to nvidia-smi subprocess calls.
    Spawns a lightweight background thread that polls GPU power/memory at a
    configurable interval while a sample is being processed.
    """

    def __init__(self, gpu_index: int = 0, poll_interval: float = 0.05):
        self.gpu_index = gpu_index
        self.poll_interval = poll_interval
        self._handle = None
        if PYNVML_AVAILABLE:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        # Per-sample scratch (reset each start_sample)
        self._sample_power_readings: List[float] = []
        self._sample_vram_readings: List[float] = []
        self._sample_start: float = 0.0
        self._gen_start: float = 0.0
        self._gen_elapsed: float = 0.0
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None

        # Aggregate accumulators
        self.sample_total_times: List[float] = []
        self.sample_gen_times: List[float] = []
        self.sample_num_tokens: List[int] = []
        self.sample_energy: List[float] = []
        self.sample_avg_power: List[float] = []
        self.sample_peak_vram: List[float] = []

    # ---- low-level GPU readers -------------------------------------------

    def _read_power_watts(self) -> Optional[float]:
        """Return current GPU power draw in watts."""
        if self._handle is not None:
            try:
                return pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            except pynvml.NVMLError:
                return None
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 f"--id={self.gpu_index}",
                 "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                timeout=2,
            )
            return float(out.decode().strip())
        except Exception:
            return None

    def _read_vram_mb(self) -> Optional[float]:
        """Return current GPU memory used in MiB."""
        if self._handle is not None:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return info.used / (1024 ** 2)
            except pynvml.NVMLError:
                return None
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 f"--id={self.gpu_index}",
                 "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                timeout=2,
            )
            return float(out.decode().strip())
        except Exception:
            return None

    # ---- background poller -----------------------------------------------

    def _poll_loop(self):
        while self._polling:
            pw = self._read_power_watts()
            vr = self._read_vram_mb()
            if pw is not None:
                self._sample_power_readings.append(pw)
            if vr is not None:
                self._sample_vram_readings.append(vr)
            time.sleep(self.poll_interval)

    # ---- public API -------------------------------------------------------

    def start_sample(self):
        """Call at the very beginning of a sample (before retrieval/tokenization)."""
        self._sample_power_readings = []
        self._sample_vram_readings = []
        self._gen_elapsed = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._sample_start = time.perf_counter()
        self._polling = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def start_generation(self):
        """Call right before model.generate()."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._gen_start = time.perf_counter()

    def end_generation(self, num_new_tokens: int = 0):
        """Call right after model.generate(). Pass number of generated tokens."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._gen_elapsed = time.perf_counter() - self._gen_start
        self.sample_gen_times.append(self._gen_elapsed)
        self.sample_num_tokens.append(num_new_tokens)

    def end_sample(self):
        """Call at the very end of a sample (after decoding)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_elapsed = time.perf_counter() - self._sample_start
        self._polling = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)

        self.sample_total_times.append(total_elapsed)

        if self._sample_power_readings:
            avg_pw = sum(self._sample_power_readings) / len(self._sample_power_readings)
            energy_j = avg_pw * total_elapsed
        else:
            avg_pw = 0.0
            energy_j = 0.0
        self.sample_avg_power.append(avg_pw)
        self.sample_energy.append(energy_j)

        peak_vram = max(self._sample_vram_readings) if self._sample_vram_readings else 0.0
        self.sample_peak_vram.append(peak_vram)

    def summary(self) -> Dict:
        """Return aggregate inference statistics."""
        n = len(self.sample_total_times)
        if n == 0:
            return {}

        total_time = sum(self.sample_total_times)
        total_gen_time = sum(self.sample_gen_times) if self.sample_gen_times else 0.0
        total_tokens = sum(self.sample_num_tokens) if self.sample_num_tokens else 0

        # Throughput: generated tokens / generation-only wall time
        throughput = (total_tokens / total_gen_time) if total_gen_time > 0 else 0.0

        # Inference latency (per paper convention): avg generation time per sample
        # i.e. time the model spends producing output for one input
        avg_latency = (total_gen_time / n) if n > 0 else 0.0

        return {
            "num_samples": n,
            # End-to-end (retrieval + tokenization + generation + decoding)
            "total_time_s": round(total_time, 3),
            "avg_time_per_sample_s": round(total_time / n, 4),
            # Generation only (model.generate)
            "total_generation_time_s": round(total_gen_time, 3),
            "avg_generation_time_per_sample_s": round(total_gen_time / n, 4) if n > 0 else 0.0,
            # Inference latency = avg generation time per sample (common in papers)
            "inference_latency_s": round(avg_latency, 4),
            # Token throughput
            "total_tokens_generated": total_tokens,
            "throughput_tokens_per_s": round(throughput, 2),
            # Energy
            "total_energy_j": round(sum(self.sample_energy), 3),
            "avg_energy_per_sample_j": round(sum(self.sample_energy) / n, 4),
            # Power
            "avg_power_w": round(sum(self.sample_avg_power) / n, 2),
            # VRAM
            "peak_vram_mb": round(max(self.sample_peak_vram), 2) if self.sample_peak_vram else 0.0,
            "avg_peak_vram_per_sample_mb": round(sum(self.sample_peak_vram) / n, 2),
            # Per-sample detail
            "per_sample_total_time_s": [round(t, 4) for t in self.sample_total_times],
            "per_sample_generation_time_s": [round(t, 4) for t in self.sample_gen_times],
            "per_sample_num_tokens": list(self.sample_num_tokens),
            "per_sample_energy_j": [round(e, 4) for e in self.sample_energy],
            "per_sample_avg_power_w": [round(p, 2) for p in self.sample_avg_power],
            "per_sample_peak_vram_mb": [round(v, 2) for v in self.sample_peak_vram],
        }


# ---- mamba_ssm imports (required for state-spaces/mamba2-* models) ----
MAMBA_SSM_OK = False
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig as _MambaConfig
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
    from dataclasses import fields as dataclass_fields
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
# CONTRIEVER RETRIEVAL MODULE (same as finetune_lamp_lora.py)
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


# =============================================================================
# QUERY/CORPUS MAKERS FOR EACH TASK (same as finetune_lamp_lora.py)
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
    """LaMP-1: Citation classification. Profile has: title, abstract, id"""
    corpus = [f'{x["title"]} {x.get("abstract", "")}' for x in profile]
    extracted = extract_strings_between_quotes(inp)
    query = f'{extracted[1]} {extracted[2]}' if len(extracted) >= 3 else inp
    return corpus, query


def lamp2_query_corpus(inp: str, profile: List[Dict]) -> Tuple[List[str], str]:
    """LaMP-2: Movie tag classification. Profile has: description, tag, id"""
    corpus = [f'{x.get("description", "")}' for x in profile]
    query = extract_after_keyword(inp, 'description:')
    return corpus, query


def lamp3_query_corpus(inp: str, profile: List[Dict]) -> Tuple[List[str], str]:
    """LaMP-3: Product rating prediction. Profile has: text, score, id"""
    corpus = [f'{x.get("text", "")}' for x in profile]
    query = extract_after_keyword(inp, 'review:')
    return corpus, query


def lamp4_query_corpus(inp: str, profile: List[Dict]) -> Tuple[List[str], str]:
    """LaMP-4: News headline generation. Profile has: title, text, id"""
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
# PROMPT GENERATORS FOR EACH TASK (same as finetune_lamp_lora.py)
# =============================================================================

def add_string_after_title(original_string: str, string_to_add: str) -> str:
    """Add context string after 'title' keyword."""
    title_index = original_string.find("title")
    if title_index == -1:
        return string_to_add + " " + original_string
    return original_string[:title_index+5] + ", and " + string_to_add + original_string[title_index+5:]


def create_lamp1_prompt(inp: str, profile: List[Dict], max_length: int, tokenizer) -> str:
    """LaMP-1: Citation classification prompt. Profile has: title, abstract, id"""
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
# DATA LOADING
# =============================================================================

def load_data(data_dir: str) -> List[Dict]:
    """Load validation data with profiles."""
    inputs_path = os.path.join(data_dir, "validation", "inputs.json")
    outputs_path = os.path.join(data_dir, "validation", "outputs.json")
    
    with open(inputs_path, 'r', encoding='utf-8') as f:
        inputs_data = json.load(f)
    
    with open(outputs_path, 'r', encoding='utf-8') as f:
        outputs_data = json.load(f)
    
    if isinstance(outputs_data, dict) and 'golds' in outputs_data:
        outputs_data = outputs_data['golds']
    
    output_lookup = {item['id']: item['output'] for item in outputs_data}
    
    merged = []
    for item in inputs_data:
        if item['id'] in output_lookup:
            merged.append({
                'id': item['id'],
                'input': item['input'],
                'profile': item.get('profile', []),  # Include profile data
                'output': output_lookup[item['id']]
            })
    return merged


def create_augmented_prompt(
    item: Dict, 
    task: str, 
    tokenizer, 
    retriever: Optional[ContrieverRetriever],
    num_retrieved: int = 4,
    max_length: int = 512,
    max_profile_size: int = 200
) -> str:
    """Create profile-augmented prompt for evaluation (same as training)."""
    inp = item['input']
    profile = item.get('profile', [])
    
    if not profile:
        return inp
    
    # Cap profile size to avoid slowdowns
    if len(profile) > max_profile_size:
        profile = profile[:max_profile_size]
    
    query_corpus_maker = QUERY_CORPUS_MAKERS.get(task)
    prompt_creator = PROMPT_CREATORS.get(task)
    
    if not query_corpus_maker or not prompt_creator:
        return inp
    
    # Retrieve top-k profile items using Contriever
    if retriever and profile:
        corpus, query = query_corpus_maker(inp, profile)
        selected_profile = retriever.retrieve_top_k(corpus, profile, query, num_retrieved)
    else:
        selected_profile = profile[:num_retrieved] if profile else []
    
    # Generate prompt with profile context
    factor = 0.6
    while factor > 0:
        try:
            max_len_prompt = max_length - min(
                len(tokenizer(inp)['input_ids']), 
                int(factor * max_length)
            )
            source = prompt_creator(inp, selected_profile, max_len_prompt, tokenizer)
            break
        except:
            factor -= 0.1
            if factor <= 0:
                source = inp
    
    return source


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def generate_predictions_causal(
    model, 
    tokenizer, 
    data: List[Dict], 
    task: str,
    retriever: Optional[ContrieverRetriever] = None,
    num_retrieved: int = 4,
    max_length: int = 512,
    max_new_tokens: int = 32,
    max_profile_size: int = 200,
    tracker: Optional[GPUInferenceTracker] = None
):
    """Generate predictions for causal LM models (TinyLlama)."""
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for item in tqdm(data, desc="Generating predictions (Causal LM)"):
            if tracker:
                tracker.start_sample()
            
            augmented_input = create_augmented_prompt(
                item, task, tokenizer, retriever, num_retrieved, max_length, max_profile_size
            )
            prompt = f"<s>[INST] {augmented_input} [/INST]"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            if tracker:
                tracker.start_generation()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            if tracker:
                tracker.end_generation(num_new_tokens=len(generated_ids))
            
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            if tracker:
                tracker.end_sample()
            
            predictions.append({
                'id': item['id'],
                'prediction': prediction,
                'ground_truth': item['output']
            })
    
    return predictions


def generate_predictions_mamba2(
    model, 
    tokenizer, 
    data: List[Dict], 
    task: str,
    retriever: Optional[ContrieverRetriever] = None,
    num_retrieved: int = 4,
    max_length: int = 512,
    max_new_tokens: int = 32,
    max_profile_size: int = 200,
    tracker: Optional[GPUInferenceTracker] = None
):
    """Generate predictions for Mamba2 models."""
    predictions = []
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for item in tqdm(data, desc="Generating predictions (Mamba2)"):
            if tracker:
                tracker.start_sample()
            
            augmented_input = create_augmented_prompt(
                item, task, tokenizer, retriever, num_retrieved, max_length, max_profile_size
            )
            prompt = f"{augmented_input}\nAnswer:"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if tracker:
                tracker.start_generation()
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=max_new_tokens,
            )
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            if tracker:
                tracker.end_generation(num_new_tokens=len(generated_ids))
            
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            if tracker:
                tracker.end_sample()
            
            predictions.append({
                'id': item['id'],
                'prediction': prediction,
                'ground_truth': item['output']
            })
    
    return predictions


def generate_predictions_qwen3(
    model, 
    tokenizer, 
    data: List[Dict], 
    task: str,
    retriever: Optional[ContrieverRetriever] = None,
    num_retrieved: int = 4,
    max_length: int = 512,
    max_new_tokens: int = 32,
    max_profile_size: int = 200,
    tracker: Optional[GPUInferenceTracker] = None
):
    """Generate predictions for Qwen3 models."""
    predictions = []
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for item in tqdm(data, desc="Generating predictions (Qwen3)"):
            if tracker:
                tracker.start_sample()
            
            augmented_input = create_augmented_prompt(
                item, task, tokenizer, retriever, num_retrieved, max_length, max_profile_size
            )
            prompt = f"<|im_start|>user\n{augmented_input}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if tracker:
                tracker.start_generation()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            if tracker:
                tracker.end_generation(num_new_tokens=len(generated_ids))
            
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            if tracker:
                tracker.end_sample()
            
            predictions.append({
                'id': item['id'],
                'prediction': prediction,
                'ground_truth': item['output']
            })
    
    return predictions


def generate_predictions_mamba1(
    model, 
    tokenizer, 
    data: List[Dict], 
    task: str,
    retriever: Optional[ContrieverRetriever] = None,
    num_retrieved: int = 4,
    max_length: int = 512,
    max_new_tokens: int = 32,
    max_profile_size: int = 200,
    tracker: Optional[GPUInferenceTracker] = None
):
    """Generate predictions for Mamba-1 1.4B models (HuggingFace-compatible)."""
    predictions = []
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for item in tqdm(data, desc="Generating predictions (Mamba-1)"):
            if tracker:
                tracker.start_sample()
            
            augmented_input = create_augmented_prompt(
                item, task, tokenizer, retriever, num_retrieved, max_length, max_profile_size
            )
            prompt = f"{augmented_input}\nAnswer:"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if tracker:
                tracker.start_generation()
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            if tracker:
                tracker.end_generation(num_new_tokens=len(generated_ids))
            
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            if tracker:
                tracker.end_sample()
            
            predictions.append({
                'id': item['id'],
                'prediction': prediction,
                'ground_truth': item['output']
            })
    
    return predictions


def generate_predictions_seq2seq(
    model, 
    tokenizer, 
    data: List[Dict], 
    task: str,
    retriever: Optional[ContrieverRetriever] = None,
    num_retrieved: int = 4,
    max_length: int = 512,
    max_new_tokens: int = 32,
    max_profile_size: int = 200,
    tracker: Optional[GPUInferenceTracker] = None
):
    """Generate predictions for seq2seq models (Flan-T5)."""
    predictions = []
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for item in tqdm(data, desc="Generating predictions (Seq2Seq)"):
            if tracker:
                tracker.start_sample()
            
            augmented_input = create_augmented_prompt(
                item, task, tokenizer, retriever, num_retrieved, max_length, max_profile_size
            )
            
            inputs = tokenizer(
                augmented_input, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if tracker:
                tracker.start_generation()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
            )
            if tracker:
                tracker.end_generation(num_new_tokens=len(outputs[0]))
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            if tracker:
                tracker.end_sample()
            
            predictions.append({
                'id': item['id'],
                'prediction': prediction,
                'ground_truth': item['output']
            })
    
    return predictions


def compute_metrics_classification(predictions: List[Dict]) -> Dict:
    """Compute accuracy and F1 for classification tasks (lamp1, lamp2)."""
    
    # Direct comparison - normalize both pred and gold
    preds = [p['prediction'].strip().lower() for p in predictions]
    golds = [p['ground_truth'].strip().lower() for p in predictions]
    
    # Get unique labels from ground truth
    unique_labels = sorted(set(golds))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    # Convert to indices
    pred_indices = []
    gold_indices = []
    
    for pred, gold in zip(preds, golds):
        gold_idx = label_to_idx.get(gold, -1)
        
        # Try to match prediction to a known label
        pred_idx = label_to_idx.get(pred, -1)
        
        # If no exact match, try partial matching
        if pred_idx == -1:
            for label, idx in label_to_idx.items():
                if label in pred or pred in label:
                    pred_idx = idx
                    break
        
        pred_indices.append(pred_idx)
        gold_indices.append(gold_idx)
    
    # Compute metrics
    correct = sum(1 for p, g in zip(pred_indices, gold_indices) if p == g and p != -1)
    total = len(predictions)
    valid = sum(1 for p in pred_indices if p != -1)
    
    accuracy = correct / total if total > 0 else 0.0
    
    # For F1, only use valid predictions
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
    
    # Show sample comparisons
    print("\nSample predictions:")
    for i, p in enumerate(predictions[:10]):
        match = "✓" if preds[i] == golds[i] else "✗"
        print(f"  {match} Pred: '{p['prediction']}' | Gold: '{p['ground_truth']}'")
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "correct": correct,
        "total_samples": total,
        "valid_predictions": valid,
        "unique_labels": unique_labels
    }


def compute_metrics_regression(predictions: List[Dict]) -> Dict:
    """Compute RMSE and MAE for regression tasks (lamp3 - rating prediction)."""
    
    pred_scores = []
    gold_scores = []
    invalid_count = 0
    
    for p in predictions:
        gold_str = p['ground_truth'].strip()
        pred_str = p['prediction'].strip()
        
        # Try to extract numeric score from prediction
        try:
            gold_score = float(gold_str)
        except ValueError:
            # If gold can't be parsed, skip this sample
            continue
        
        try:
            # Try direct float conversion first
            pred_score = float(pred_str)
        except ValueError:
            # Try to extract first number from prediction
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', pred_str)
            if numbers:
                pred_score = float(numbers[0])
            else:
                # If no number found, count as invalid and skip
                invalid_count += 1
                continue
        
        # Clamp to valid rating range (1-5) for lamp3
        pred_score = max(1.0, min(5.0, pred_score))
        
        pred_scores.append(pred_score)
        gold_scores.append(gold_score)
    
    if not pred_scores:
        print("\nWarning: No valid numeric predictions found!")
        return {
            "rmse": float('inf'),
            "mae": float('inf'),
            "total_samples": len(predictions),
            "valid_predictions": 0,
            "invalid_predictions": invalid_count
        }
    
    # Compute RMSE and MAE
    mse = mean_squared_error(gold_scores, pred_scores)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(gold_scores, pred_scores)
    
    # Show sample comparisons
    print("\nSample predictions:")
    for i, p in enumerate(predictions[:10]):
        gold_str = p['ground_truth'].strip()
        pred_str = p['prediction'].strip()
        try:
            gold_val = float(gold_str)
            # Try to extract prediction value
            try:
                pred_val = float(pred_str)
            except ValueError:
                import re
                numbers = re.findall(r'[-+]?\d*\.?\d+', pred_str)
                pred_val = float(numbers[0]) if numbers else None
            
            if pred_val is not None:
                diff = abs(pred_val - gold_val)
                match = "✓" if diff < 0.5 else "✗"
                print(f"  {match} Pred: {pred_val:.1f} | Gold: {gold_val:.1f} | Diff: {diff:.2f}")
            else:
                print(f"  ✗ Pred: '{pred_str}' (invalid) | Gold: {gold_val:.1f}")
        except:
            print(f"  ? Pred: '{pred_str}' | Gold: '{gold_str}'")
    
    return {
        "rmse": rmse,
        "mae": mae,
        "total_samples": len(predictions),
        "valid_predictions": len(pred_scores),
        "invalid_predictions": invalid_count
    }


def compute_metrics_rouge(predictions: List[Dict]) -> Dict:
    """Compute ROUGE-1 and ROUGE-L for text generation tasks (lamp4 - headline generation)."""
    
    if not ROUGE_AVAILABLE:
        print("\nError: rouge_score package not installed!")
        print("Install with: pip install rouge-score")
        return {
            "rouge1": 0.0,
            "rougeL": 0.0,
            "total_samples": len(predictions),
            "error": "rouge_score not installed"
        }
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rougeL_scores = []
    
    for p in predictions:
        gold = p['ground_truth'].strip()
        pred = p['prediction'].strip()
        
        if not pred:
            # Empty prediction gets 0 score
            rouge1_scores.append(0.0)
            rougeL_scores.append(0.0)
            continue
        
        scores = scorer.score(gold, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0.0
    avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0.0
    
    # Show sample comparisons
    print("\nSample predictions:")
    for i, p in enumerate(predictions[:10]):
        gold = p['ground_truth'].strip()
        pred = p['prediction'].strip()
        if pred and ROUGE_AVAILABLE:
            scores = scorer.score(gold, pred)
            r1 = scores['rouge1'].fmeasure
            rL = scores['rougeL'].fmeasure
            match = "✓" if r1 > 0.5 else "✗"
            print(f"  {match} R1:{r1:.2f} RL:{rL:.2f} | Pred: '{pred[:50]}...' | Gold: '{gold[:50]}...'")
        else:
            print(f"  ✗ Pred: '{pred[:50]}' | Gold: '{gold[:50]}'")
    
    return {
        "rouge1": avg_rouge1,
        "rougeL": avg_rougeL,
        "total_samples": len(predictions),
        "valid_predictions": len([s for s in rouge1_scores if s > 0])
    }


def compute_metrics(predictions: List[Dict], task: str) -> Dict:
    """Compute task-appropriate metrics.
    
    - lamp1, lamp2: Accuracy and F1 (classification)
    - lamp3: RMSE and MAE (regression/rating prediction)
    - lamp4: ROUGE-1 and ROUGE-L (text generation)
    """
    if task in ["lamp1", "lamp2"]:
        return compute_metrics_classification(predictions)
    elif task == "lamp3":
        return compute_metrics_regression(predictions)
    elif task == "lamp4":
        return compute_metrics_rouge(predictions)
    else:
        # Default to classification metrics
        return compute_metrics_classification(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to fine-tuned LoRA adapter (not required for --zero_shot)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--task", type=str, required=True, choices=["lamp1", "lamp2", "lamp3", "lamp4"])
    parser.add_argument("--model_type", type=str, required=True, choices=["flan-t5", "tinyllama", "mamba2", "mamba1", "qwen3", "qwen"],
                        help="Model type: 'flan-t5', 'tinyllama', 'mamba2', 'mamba1', or 'qwen3' (alias: 'qwen')")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name (default: auto-select based on model_type)")
    parser.add_argument("--method", type=str, default="lora", choices=["lora", "loraplus", "full_ft", "bitfit", "qlora"],
                        help="Training method used: 'lora' (default), 'loraplus', 'full_ft', 'bitfit', or 'qlora'. "
                             "LoRA/LoRA+/QLoRA load a PEFT adapter; full_ft/bitfit load the complete model from model_path.")
    parser.add_argument("--qlora_bits", type=int, default=4, choices=[4, 8],
                        help="Quantization bits for QLoRA evaluation (default: 4)")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Evaluate base model zero-shot (no LoRA adapter, no fine-tuning)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (defaults to model_path; required for --zero_shot)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick testing")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--num_retrieved", type=int, default=4, help="Number of profile items to retrieve")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_profile_size", type=int, default=200,
                        help="Max profile items to consider for retrieval (default: 200)")
    parser.add_argument("--use_retrieval", action="store_true", default=True,
                        help="Use Contriever retrieval (default: True)")
    parser.add_argument("--no_retrieval", action="store_false", dest="use_retrieval",
                        help="Disable Contriever retrieval")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 for Mamba2 (default)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 instead of BF16")
    args = parser.parse_args()
    
    if args.model_type == "qwen":
        args.model_type = "qwen3"
    
    # Validate arguments
    if not args.zero_shot and args.model_path is None:
        parser.error("--model_path is required unless --zero_shot is used")
    
    # Determine output directory for saving results
    if args.output_dir is not None:
        save_dir = args.output_dir
    elif args.model_path is not None:
        save_dir = args.model_path
    else:
        save_dir = os.path.join(".", f"zero_shot_{args.model_type}_{args.task}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Set default base model
    if args.base_model is None:
        if args.model_type == "flan-t5":
            args.base_model = "google/flan-t5-base"
        elif args.model_type == "mamba2":
            args.base_model = "state-spaces/mamba2-1.3b"
        elif args.model_type == "mamba1":
            args.base_model = "state-spaces/mamba-1.4b-hf"
        elif args.model_type == "qwen3":
            args.base_model = "Qwen/Qwen3-1.7B"
        else:  # tinyllama
            args.base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Check mamba_ssm availability for mamba2
    if args.model_type == "mamba2" and not MAMBA_SSM_OK:
        raise ImportError(
            "mamba_ssm package is required for mamba2 models.\n"
            "Install with: pip install mamba-ssm\n"
            "Also requires: pip install causal-conv1d>=1.2.0"
        )
    
    method_names = {"lora": "LoRA", "loraplus": "LoRA+", "full_ft": "Full-FT", "bitfit": "BitFit", "qlora": "QLoRA"}
    use_full_ft = args.method == "full_ft"
    use_bitfit = args.method == "bitfit"
    use_qlora = args.method == "qlora"
    loads_full_model = use_full_ft or use_bitfit
    loads_adapter = not loads_full_model and not args.zero_shot
    
    if use_qlora and not BNB_AVAILABLE:
        raise ImportError("QLoRA requires bitsandbytes. Install with: pip install bitsandbytes")
    
    if args.zero_shot:
        eval_mode = "Zero-Shot"
    else:
        eval_mode = f"Fine-Tuned ({method_names.get(args.method, args.method)})"
    print(f"\n{'='*50}")
    print(f"{eval_mode} Evaluation")
    print(f"{'='*50}")
    print(f"Model type: {args.model_type}")
    print(f"Base model: {args.base_model}")
    if args.zero_shot:
        print(f"Mode: ZERO-SHOT (no adapter)")
    elif loads_full_model:
        label = "BitFit" if use_bitfit else "FULL-FT"
        print(f"Mode: {label} (loading complete model from checkpoint)")
        print(f"Model checkpoint: {args.model_path}")
    elif use_qlora:
        print(f"Mode: QLoRA ({args.qlora_bits}-bit quantization + LoRA adapter)")
        print(f"Adapter: {args.model_path}")
    else:
        print(f"Mode: {method_names.get(args.method, args.method)} (loading adapter)")
        print(f"LoRA adapter: {args.model_path}")
    print(f"Task: {args.task}")
    print(f"Profile retrieval: {'enabled' if args.use_retrieval else 'disabled'}")
    print(f"Num retrieved: {args.num_retrieved}")
    print(f"Results will be saved to: {save_dir}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Initialize retriever
    retriever = None
    if args.use_retrieval:
        print("\nLoading Contriever retriever...")
        retriever = ContrieverRetriever(device=device_str)
    
    # Build BitsAndBytes config for QLoRA
    bnb_config = None
    if use_qlora:
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(args.qlora_bits == 4),
            load_in_8bit=(args.qlora_bits == 8),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    
    if args.model_type == "mamba2":
        # Mamba2 uses EleutherAI/gpt-neox-20b tokenizer
        print("\nLoading tokenizer (EleutherAI/gpt-neox-20b for Mamba2)...")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        print(f"   Using tokenizer: EleutherAI/gpt-neox-20b (padding: {tokenizer.padding_side})")
        
        # Load model: full_ft/bitfit loads from checkpoint, otherwise from base_model
        model_source = args.model_path if loads_full_model else args.base_model
        print(f"Loading Mamba2 backbone via mamba_ssm from {model_source}...")
        _dtype = torch.bfloat16 if not args.fp16 else torch.float16
        _config_data = load_config_hf(model_source)
        _valid_keys = {f.name for f in dataclass_fields(_MambaConfig)}
        _filtered = {k: v for k, v in _config_data.items() if k in _valid_keys}
        _cfg = _MambaConfig(**_filtered)
        mamba_backbone = MambaLMHeadModel(_cfg, device=device, dtype=_dtype)
        _state_dict = load_state_dict_hf(model_source, device=device, dtype=_dtype)
        _prefix = "backbone."
        if any(k.startswith(_prefix) for k in _state_dict):
            _state_dict = {
                k[len(_prefix):] if k.startswith(_prefix) else k: v
                for k, v in _state_dict.items()
            }
        mamba_backbone.load_state_dict(_state_dict)
        
        mamba_config = mamba_backbone.config
        print(f"   Config: d_model={mamba_config.d_model}, n_layer={mamba_config.n_layer}")
        config = Mamba2ConfigWrapper(mamba_config)
        model = Mamba2ForCausalLM(config, backbone=mamba_backbone)
        
        if not args.zero_shot and not loads_full_model:
            adapter_label = "QLoRA" if use_qlora else "LoRA"
            print(f"Loading {adapter_label} adapter from {args.model_path}...")
            model = PeftModel.from_pretrained(model, args.model_path)
        model.eval()
    elif args.model_type == "mamba1":
        # Mamba-1 1.4B: HuggingFace-compatible model (no mamba_ssm needed)
        print(f"\nLoading Mamba-1 tokenizer from {args.base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        print(f"   Using Mamba-1 tokenizer (padding: {tokenizer.padding_side})")
        
        model_source = args.model_path if loads_full_model else args.base_model
        print(f"Loading Mamba-1 model from {model_source}...")
        mamba1_load_kwargs = dict(
            torch_dtype=torch.bfloat16 if not args.fp16 else torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        if use_qlora and bnb_config is not None:
            mamba1_load_kwargs["quantization_config"] = bnb_config
        model = AutoModelForCausalLM.from_pretrained(model_source, **mamba1_load_kwargs)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        
        if not args.zero_shot and not loads_full_model:
            adapter_label = "QLoRA" if use_qlora else "LoRA"
            print(f"Loading {adapter_label} adapter from {args.model_path}...")
            model = PeftModel.from_pretrained(model, args.model_path)
        model.eval()
    elif args.model_type == "qwen3":
        # Qwen3 model loading
        print(f"\nLoading Qwen3 tokenizer from {args.base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        print(f"   Using Qwen3 tokenizer (padding: {tokenizer.padding_side})")
        
        model_source = args.model_path if loads_full_model else args.base_model
        print(f"Loading Qwen3 model from {model_source}...")
        qwen3_load_kwargs = dict(
            torch_dtype=torch.bfloat16 if not args.fp16 else torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        if use_qlora and bnb_config is not None:
            qwen3_load_kwargs["quantization_config"] = bnb_config
        model = AutoModelForCausalLM.from_pretrained(model_source, **qwen3_load_kwargs)
        model.config.pad_token_id = tokenizer.pad_token_id
        
        if not args.zero_shot and not loads_full_model:
            adapter_label = "QLoRA" if use_qlora else "LoRA"
            print(f"Loading {adapter_label} adapter from {args.model_path}...")
            model = PeftModel.from_pretrained(model, args.model_path)
        model.eval()
    else:
        print(f"\nLoading tokenizer from {args.base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        
        if args.model_type == "flan-t5":
            model_source = args.model_path if loads_full_model else args.base_model
            print(f"Loading Flan-T5 model from {model_source}...")
            t5_load_kwargs = dict(
                torch_dtype=torch.float16,
                device_map="auto",
            )
            if use_qlora and bnb_config is not None:
                t5_load_kwargs["quantization_config"] = bnb_config
            model = AutoModelForSeq2SeqLM.from_pretrained(model_source, **t5_load_kwargs)
        else:  # tinyllama
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model_source = args.model_path if loads_full_model else args.base_model
            print(f"Loading TinyLlama model from {model_source}...")
            llama_load_kwargs = dict(
                torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float16),
                device_map="auto",
            )
            if use_qlora and bnb_config is not None:
                llama_load_kwargs["quantization_config"] = bnb_config
            model = AutoModelForCausalLM.from_pretrained(model_source, **llama_load_kwargs)
        
        if not args.zero_shot and not loads_full_model:
            adapter_label = "QLoRA" if use_qlora else "LoRA"
            print(f"Loading {adapter_label} adapter from {args.model_path}...")
            model = PeftModel.from_pretrained(model, args.model_path)
        model.eval()
    
    print(f"\nLoading validation data from {args.data_dir}...")
    data = load_data(args.data_dir)
    print(f"Loaded {len(data)} samples")
    
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Using first {args.max_samples} samples")
    
    # Initialize inference tracker
    gpu_idx = 0
    if torch.cuda.is_available():
        gpu_idx = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    tracker = GPUInferenceTracker(gpu_index=gpu_idx)
    print(f"Inference tracking: enabled (pynvml={'yes' if PYNVML_AVAILABLE else 'no, falling back to nvidia-smi'})")
    
    print("\nGenerating predictions...")
    if args.model_type == "flan-t5":
        predictions = generate_predictions_seq2seq(
            model, tokenizer, data, args.task,
            retriever=retriever,
            num_retrieved=args.num_retrieved,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            max_profile_size=args.max_profile_size,
            tracker=tracker
        )
    elif args.model_type == "mamba2":
        predictions = generate_predictions_mamba2(
            model, tokenizer, data, args.task,
            retriever=retriever,
            num_retrieved=args.num_retrieved,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            max_profile_size=args.max_profile_size,
            tracker=tracker
        )
    elif args.model_type == "mamba1":
        predictions = generate_predictions_mamba1(
            model, tokenizer, data, args.task,
            retriever=retriever,
            num_retrieved=args.num_retrieved,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            max_profile_size=args.max_profile_size,
            tracker=tracker
        )
    elif args.model_type == "qwen3":
        predictions = generate_predictions_qwen3(
            model, tokenizer, data, args.task,
            retriever=retriever,
            num_retrieved=args.num_retrieved,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            max_profile_size=args.max_profile_size,
            tracker=tracker
        )
    else:  # tinyllama
        predictions = generate_predictions_causal(
            model, tokenizer, data, args.task,
            retriever=retriever,
            num_retrieved=args.num_retrieved,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            max_profile_size=args.max_profile_size,
            tracker=tracker
        )
    
    # Collect inference stats
    inference_stats = tracker.summary()
    
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, args.task)
    
    print("\n" + "="*50)
    eval_label = "ZERO-SHOT" if args.zero_shot else "FINE-TUNED"
    print(f"{eval_label} EVALUATION RESULTS FOR {args.task.upper()}")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    if inference_stats:
        print("\n" + "="*50)
        print("INFERENCE STATISTICS")
        print("="*50)
        print("  --- Timing ---")
        print(f"  Total end-to-end time:           {inference_stats['total_time_s']:.3f} s")
        print(f"  Avg end-to-end per sample:       {inference_stats['avg_time_per_sample_s']:.4f} s")
        print(f"  Total generation time:           {inference_stats['total_generation_time_s']:.3f} s")
        print(f"  Avg generation per sample:       {inference_stats['avg_generation_time_per_sample_s']:.4f} s")
        print(f"  Inference latency (avg gen/sample): {inference_stats['inference_latency_s']:.4f} s")
        print("  --- Throughput ---")
        print(f"  Total tokens generated:          {inference_stats['total_tokens_generated']}")
        print(f"  Throughput:                      {inference_stats['throughput_tokens_per_s']:.2f} tokens/s")
        print("  --- Energy & Power ---")
        print(f"  Total energy consumed:           {inference_stats['total_energy_j']:.3f} J")
        print(f"  Avg energy per sample:           {inference_stats['avg_energy_per_sample_j']:.4f} J")
        print(f"  Avg GPU power draw:              {inference_stats['avg_power_w']:.2f} W")
        print("  --- VRAM ---")
        print(f"  Peak VRAM usage:                 {inference_stats['peak_vram_mb']:.2f} MiB")
        print(f"  Avg peak VRAM per sample:        {inference_stats['avg_peak_vram_per_sample_mb']:.2f} MiB")
    
    # Save results
    result_prefix = "zero_shot_" if args.zero_shot else ""
    results_path = os.path.join(save_dir, f"{result_prefix}eval_results.json")
    save_data = {
        "mode": "zero_shot" if args.zero_shot else args.method,
        "model_type": args.model_type,
        "base_model": args.base_model,
        "task": args.task,
        "metrics": metrics,
        "inference_stats": inference_stats,
    }
    if not args.zero_shot:
        save_data["model_path"] = args.model_path
        save_data["method"] = args.method
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save predictions
    preds_path = os.path.join(save_dir, f"{result_prefix}predictions.json")
    with open(preds_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {preds_path}")
    
    # Save detailed inference stats separately
    if inference_stats:
        inference_path = os.path.join(save_dir, f"{result_prefix}inference_stats.json")
        with open(inference_path, 'w') as f:
            json.dump(inference_stats, f, indent=2)
        print(f"Inference stats saved to {inference_path}")


if __name__ == "__main__":
    main()



