# LaMP Benchmark Framework

A single-script framework for fine-tuning small LLMs (Transformers, encoder-decoder, and State-Space Models) on the **LaMP** personalization tasks, using Contriever retrieval over each user's profile. It measures **GPU energy, power, VRAM, and inference latency**, estimates **FLOPs**, and scores every run with **NetScore** and its efficiency variants (**NS, NS-E, NS-M, NS#**).

This replaces the former `finetune_lamp.py` and `evaluate_lamp.py`. Both are now subcommands of [lamp_benchmark.py](lamp_benchmark.py), and the SAM metric has been replaced by NetScore. The NetScore setup matches the GLUE framework.

## Features

| | Supported |
|---|---|
| **Models** | Flan-T5-base, TinyLlama-1.1B, Qwen3-1.7B, Mamba-1 1.4B, Mamba-2 1.3B |
| **Tasks** | LaMP-1 (citation), LaMP-2 (movie tags), LaMP-3 (rating, regression), LaMP-4 (headline generation) |
| **Methods** | LoRA, LoRA+, Full fine-tuning, BitFit, QLoRA |
| **Retrieval** | Contriever (`facebook/contriever`), top-k profile items (k = 4) |
| **Tracking** | NVML power/VRAM sampling during training, per-sample inference metrics, analytical FLOPs |
| **Scoring** | NetScore variants NS / NS-E / NS-M / NS#, computed for both training cost and inference cost |

A `train` run does the following:

1. Loads the tokenizer and model, then applies the method-specific setup (adapters, freezing, or quantization).
2. Retrieves the top-k profile items for every sample and builds the profile-augmented prompts.
3. Optionally runs a **zero-shot generation pass** first (`--eval_before_training`).
4. **Fine-tunes** with the Hugging Face `Trainer` while an NVML callback samples GPU power and VRAM. Validation loss is logged each epoch.
5. Frees the optimizer state, then runs the **post-training evaluation as a tracked inference pass**: one sample at a time, covering retrieval, generation, and decoding. This pass yields both the task metrics and the latency, throughput, energy, and peak VRAM.
6. Estimates **FLOPs**, computes the **NetScore variants** for the training cost and the inference cost, saves the model or adapter, and writes JSON results.

An `evaluate` run performs steps 2, 5, and 6 on an existing checkpoint, or on the base model with `--zero_shot`. It reports only the inference-cost NetScore.

---

## Requirements

- Linux with an NVIDIA GPU. NVML is used for power readings; without it, the power- and memory-based variants are `null`.
- Python 3.10+ (tested on 3.11)
- PyTorch 2.1+ built for your CUDA version

## Installation

```bash
conda create -n llm python=3.11 -y
conda activate llm

# 1. PyTorch matching your CUDA version (example: CUDA 12.8)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Core dependencies
pip install -r requirements.txt

# 3. Optional: Mamba-2 (required for --model_type mamba2)
pip install "causal-conv1d>=1.2.0" mamba-ssm --no-build-isolation
```

## Data layout

Each task folder contains `train/` and `validation/` splits in the LaMP format:

```
data/
├── dataset1/              # lamp1
│   ├── train/
│   │   ├── inputs.json    # [{"id", "input", "profile": [...]}, ...]
│   │   └── outputs.json   # [{"id", "output"}, ...]  or  {"golds": [...]}
│   └── validation/
│       ├── inputs.json
│       └── outputs.json
├── dataset2/              # lamp2
├── dataset3/              # lamp3
└── dataset4/              # lamp4
```

- **One task:** `--data_dir` can point at the dataset folder itself (`./data/dataset2`) or at its parent (`./data`).
- **Several tasks:** `--data_dir` must be the parent folder.

---

## Usage

### Fine-tune

```bash
# TinyLlama on LaMP-2 with LoRA
python lamp_benchmark.py train --model_type tinyllama --task lamp2 --method lora \
    --data_dir ./data/dataset2 --output_dir ./outputs/tinyllama_lamp2_lora --bf16

# Qwen3 on LaMP-3 with LoRA+ (lr_B = 16 x lr_A)
python lamp_benchmark.py train --model_type qwen3 --task lamp3 --method loraplus --loraplus_ratio 16 \
    --data_dir ./data/dataset3 --output_dir ./outputs/qwen3_lamp3_loraplus --bf16

# QLoRA (4-bit NF4) and also measure the zero-shot baseline first
python lamp_benchmark.py train --model_type tinyllama --task lamp1 --method qlora --qlora_bits 4 \
    --data_dir ./data/dataset1 --output_dir ./outputs/tinyllama_lamp1_qlora --bf16 --eval_before_training

# Quick smoke test: 8 train / 4 validation samples, 1 epoch
python lamp_benchmark.py train --model_type flan-t5 --task lamp1 --method lora \
    --data_dir ./data/dataset1 --output_dir ./outputs/debug --debug
```

### Sweeps

`--model_type`, `--task`, and `--method` each accept a single value, a comma-separated list, or `all`.

- The runs are the Cartesian product of the three lists, executed one after another.
- For each dimension with more than one value, a sub-folder is added under `--output_dir`, in the order `<model_type>/<task>/<method>`.
- A failed run is logged and the sweep continues.
- At the end, a summary table lists the NetScore variants, and `all_results_<command>.json` is written.

```bash
python lamp_benchmark.py train --model_type qwen3 --task all --method lora,loraplus \
    --data_dir ./data --output_dir ./outputs/qwen3 --bf16

# Preview the planned runs, data folders, and output folders without executing
python lamp_benchmark.py train --model_type all --task all --method all \
    --data_dir ./data --output_dir ./outputs --dry_run
```

### Evaluate

```bash
# LoRA / LoRA+ / QLoRA adapter (loaded on top of the base model)
python lamp_benchmark.py evaluate --model_type tinyllama --task lamp2 --method lora \
    --data_dir ./data/dataset2 --model_path ./outputs/tinyllama_lamp2_lora/final_model --bf16

# Full fine-tuning / BitFit checkpoint (complete model)
python lamp_benchmark.py evaluate --model_type qwen3 --task lamp1 --method full_ft \
    --data_dir ./data/dataset1 --model_path ./outputs/qwen3_lamp1_full_ft/final_model --bf16

# Zero-shot base model
python lamp_benchmark.py evaluate --zero_shot --model_type mamba1 --task lamp3 \
    --data_dir ./data/dataset3 --output_dir ./zero_shot_results --bf16

# Re-evaluate every checkpoint of a training sweep (uses <run_dir>/final_model)
python lamp_benchmark.py evaluate --model_type qwen3 --task all --method lora,loraplus \
    --data_dir ./data --output_dir ./outputs/qwen3 --bf16
```

Where evaluation results are written:

- **`--output_dir` given:** results go there.
- **`--model_path` ends in `final_model`:** results go to its parent, next to the training results.
- **Other `--model_path`:** results go into the checkpoint directory.
- **Zero-shot without `--output_dir`:** results go to `./zero_shot_<model>_<task>`.

### CLI reference

Shared by `train` and `evaluate`:

| Argument | Default | Description |
|---|---|---|
| `--model_type` | *required* | `flan-t5`, `tinyllama`, `qwen3` (alias `qwen`), `mamba1`, `mamba2`, list, or `all` |
| `--model_name` / `--base_model` | per model | Override the HF model id (single model type only) |
| `--task` | *required* | `lamp1`…`lamp4` (`LaMP-1` spelling accepted), list, or `all` |
| `--method` | `lora` | `lora`, `loraplus`, `full_ft`, `bitfit`, `qlora`, list, or `all` |
| `--data_dir` | *required* | Dataset folder, or the parent of `datasetN/` |
| `--output_dir` | *required for train* | Output directory |
| `--num_retrieved` | `4` | Profile items retrieved per sample |
| `--max_profile_size` | `100` | Max profile items considered for retrieval |
| `--no_retrieval` | off | Skip Contriever and use the first `num_retrieved` profile items |
| `--max_length` | `512` | Max prompt / training sequence length (also S for FLOPs) |
| `--max_new_tokens` | `32` | Generated tokens (also T for FLOPs) |
| `--num_beams` | `4` | Beam width for Flan-T5 (causal models decode greedily) |
| `--max_samples` | all | Limit the number of validation samples |
| `--bf16` / `--fp16` | off | Weight and training precision |
| `--qlora_bits` | `4` | `4` (NF4, double quantization) or `8` |
| `--gpu_index` | `0` | GPU index used for NVML monitoring |
| `--cache_dir` | `./cache` | HF cache directory |
| `--seed` | `42` | Random seed |
| `--dry_run` | off | Print planned runs and exit |

`train` only:

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | `4` | Per-device batch size |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation |
| `--learning_rate` | `2e-4` | Base learning rate (lr_A for LoRA+) |
| `--num_epochs` | `10` | Training epochs |
| `--warmup_ratio` | `0.05` | Linear warmup ratio |
| `--weight_decay` | `0.01` | Weight decay |
| `--lora_r` / `--lora_alpha` / `--lora_dropout` | `16` / `32` / `0.1` | LoRA hyperparameters (LoRA, LoRA+, QLoRA) |
| `--loraplus_ratio` | `16` | LoRA+ learning-rate ratio lr_B / lr_A |
| `--eval_before_training` | off | Zero-shot generation pass before training; reports `performance_improvement` |
| `--no_epoch_eval` | off | Skip the per-epoch validation loss |
| `--no_save_model` | off | Do not write `final_model/` |
| `--debug` | off | 8 train and 4 validation samples, 1 epoch |
| `--nvml_sample_interval` | `100` | Background power-sampling interval in ms |
| `--nvml_sample_every_n_steps` | `10` | Also sample every N optimizer steps |
| `--nvml_no_background_sampling` | off | Disable the background sampling thread |

`evaluate` only:

| Argument | Default | Description |
|---|---|---|
| `--model_path` / `--lora_path` | — | Adapter directory (LoRA/LoRA+/QLoRA) or full checkpoint (full_ft/BitFit) |
| `--zero_shot` | off | Evaluate the base model; `--method` is ignored |

---

## Models

Defined in `MODELS` in [lamp_benchmark.py](lamp_benchmark.py).

| Key | HF checkpoint | LoRA targets | Prompt format | Notes |
|---|---|---|---|---|
| `flan-t5` | `google/flan-t5-base` | `q, v` | raw input → target | Seq2seq; beam search at evaluation |
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `q,k,v,o_proj` | `<s>[INST] … [/INST] target</s>` | |
| `qwen3` | `Qwen/Qwen3-1.7B` | `q,k,v,o_proj` | ChatML (`<|im_start|>user … assistant`) | Left padding |
| `mamba1` | `state-spaces/mamba-1.4b-hf` | `x_proj, in_proj` | `… \nAnswer: target` | Native HF Mamba, left padding |
| `mamba2` | `state-spaces/mamba2-1.3b` | `in_proj, x_proj` | `… \nAnswer: target` | Loaded via `mamba_ssm`, GPT-NeoX-20B tokenizer, left padding, saved as `.bin` |

For causal models, the whole templated text is the training target (`labels = input_ids`). Each model also has an `ArchSpec` with the shapes used by the FLOPs model; the values come from the official HF configs.

## Tasks

Defined in `TASKS` in [lamp_benchmark.py](lamp_benchmark.py). Training uses `train/`; evaluation uses `validation/`.

| Key | Folder | Task | Metrics | Primary | NetScore performance a |
|---|---|---|---|---|---|
| `lamp1` | `dataset1` | Personalized citation identification (`[1]`/`[2]`) | Accuracy, macro-F1 | Accuracy | Accuracy |
| `lamp2` | `dataset2` | Personalized movie tagging (15 tags) | Accuracy, macro-F1 | Accuracy | Accuracy |
| `lamp3` | `dataset3` | Personalized product rating (1–5) | MAE, RMSE | MAE | `max(0, 1 − MAE / 4)` |
| `lamp4` | `dataset4` | Personalized news headline generation | ROUGE-1, ROUGE-L | ROUGE-1 | ROUGE-1 |

- **Classification matching:** a prediction counts if it equals a label exactly or partially (the label contained in the prediction, or the reverse), compared case-insensitively. Empty predictions never match.
- **Rating parsing:** LaMP-3 takes the first number in the output and clamps it to [1, 5].
- **Why LaMP-3 is converted:** NetScore needs a higher-is-better score in (0, 1], so MAE is mapped using the largest possible error on a 1–5 scale (4). The divisor is `NetScoreConfig.lamp3_max_error`.

**Prompt construction** (shared by training and evaluation):

1. Contriever scores the first `max_profile_size` profile items against a task-specific query.
2. The top `num_retrieved` items are formatted with the LaMP PPEP templates. For example, LaMP-2 uses `the tag for the movie: "<description>" is "<tag>"`.
3. Each profile item's text is truncated so the whole prompt fits in `max_length`.

## Methods

| Method | What is trained | Optimizer | Notes |
|---|---|---|---|
| `lora` | LoRA adapters | AdamW (Trainer default) | `r`, `alpha`, and dropout from the CLI; `bias="none"` |
| `loraplus` | LoRA adapters, with lr_B = lr_A × ratio | PEFT `create_loraplus_optimizer` with `Adam8bit` (falls back to AdamW) | Built in `LoraPlusTrainer.create_optimizer`, so `Trainer` creates the linear warmup/decay schedule with the correct step count |
| `full_ft` | All parameters | AdamW | Use `--bf16`: fp16 weights cannot be trained with AMP fp16 |
| `bitfit` | Bias parameters | AdamW | TinyLlama, Qwen3, and T5 have no biases, so their norm weights are trained instead |
| `qlora` | LoRA adapters on a 4-bit (or 8-bit) base model | AdamW | NF4 with double quantization. **Not supported for `mamba2`**, whose `mamba_ssm` kernels don't work with bitsandbytes |

Saved checkpoints:

- **LoRA, LoRA+, QLoRA:** only the adapter is saved to `final_model/`.
- **Full fine-tuning, BitFit:** the complete model is saved there.

---

## Efficiency measurement and NetScore

### Training power and VRAM

`NVMLTrainingCallback` combines two sources and uses the first one that has data:

1. **Continuous sampling:** a background thread reads power and VRAM every `--nvml_sample_interval` ms.
2. **Step sampling:** one reading every `--nvml_sample_every_n_steps` optimizer steps.

Energy is integrated with the trapezoidal rule (`E = ∫ P dt`) and reported in Wh. The callback also records peak VRAM from NVML and the PyTorch allocator peak.

### Inference

The validation split is run **one sample at a time**. `GPUInferenceTracker` polls power and VRAM every 50 ms and records:

- **end-to-end time:** retrieval + tokenization + generation + decoding
- **generation time:** `model.generate` only; reported as latency
- **generated tokens and throughput**
- **energy:** per sample and in total
- **average power**
- **peak VRAM:** includes the Contriever model, which is part of the inference pipeline

When `pynvml` is unavailable, the tracker falls back to `nvidia-smi`.

### FLOPs

FLOPs are estimated **analytically** per sequence, with S = `max_length` prompt tokens and T = `max_new_tokens`. The LM head is included, since it isn't negligible for generation (for Qwen3 the vocabulary has 152k entries).

| Quantity | Causal (Transformer, Mamba-1/2) | Flan-T5 |
|---|---|---|
| Training forward, F_train | F(S) | encoder(S) + decoder(T) |
| Inference, F_inf | F(S + T) | encoder(S) + num_beams · decoder(T) |
| Adapter forward, A | 2 · S · trainable params | 2 · (S + T) · trainable params |
| Training: `full_ft` | 3 F_train | 3 F_train |
| Training: `bitfit` | 2 F_train | 2 F_train |
| Training: `lora`, `loraplus`, `qlora` | 2 F_train + 3A | 2 F_train + 3A |

How F is computed:

- **Transformer layers:** attention projections (GQA-aware), the attention core, and the SwiGLU MLP.
- **Mamba layers:** projections, conv, and the selective scan.
- **T5:** self-attention, cross-attention, and the gated-GELU FFN.
- **Not counted:** Contriever retrieval. Its time and energy still appear in the inference t and w.
- **Adapters at inference:** assumed merged, so they add nothing.

### NetScore

```
NetScore = S · log10( a^α / ( (p·m)^β · v^γ · t^δ · w^λ ) )        S = 20, α = 2
```

The efficiency exponents act as switches that select which cost terms a variant includes. Non-zero exponents use 1/8.

| Variant | β (p×m) | γ (VRAM) | δ (time) | λ (power) | Penalizes |
|---|---|---|---|---|---|
| `NS` | 0.5 | 0 | 0 | 0 | Model size (parameters and FLOPs) |
| `NS-E` | 0 | 0 | 0.125 | 0.125 | Energy (time and power) |
| `NS-M` | 0 | 0.125 | 0 | 0 | Peak memory |
| `NS#` | 0 | 0.125 | 0.125 | 0.125 | All efficiency terms |

Every variant is computed for each cost:

| Term | Unit | Training cost (`NetScore`) | Inference cost (`NetScore_inference`) |
|---|---|---|---|
| a | % | NetScore performance (see Tasks) × 100 | Same |
| p | millions | Trainable parameters | Same (adapter / bias / all params in `evaluate`) |
| m | millions of FLOPs | Training FLOPs per sequence | Inference FLOPs per sequence |
| v | GiB | Peak NVML VRAM during training | Peak VRAM in the inference pass |
| t | seconds | Fine-tuning wall time | End-to-end inference-pass time |
| w | W | Average power during training | Average power in the inference pass |

- **Configuration:** S, α, the variant table, the units, and the LaMP-3 error scale are set in `NetScoreConfig` in [lamp_benchmark.py](lamp_benchmark.py). Every listed variant is computed, saved, and shown in the summary table automatically.
- **Missing values:** a variant is `null` when a ≤ 0, or when one of its non-zero-exponent terms is missing or zero (for example, no NVML data).
- **Zero-shot runs:** there are no trained parameters, so p is `null`. `NS` is therefore `null`, while `NS-E`, `NS-M`, and `NS#` are still computed.
- **Inputs:** the exact inputs, in NetScore units, are saved as `netscore_inputs` and `netscore_inputs_inference`.

---

## Output structure

```
<output_dir>/                                  # + <model_type>/<task>/<method>/ per swept dimension
├── benchmark_results_<method>.json            # train: metrics, params, energy, FLOPs, NetScore (train + inference)
├── power_vram_timeseries.json                 # train: NVML step samples + summary
├── predictions.json                           # post-training predictions
├── inference_stats.json                       # aggregate + per-sample inference metrics
├── final_model/                               # adapter or full model + tokenizer (unless --no_save_model)
├── eval_results.json                          # evaluate: metrics, FLOPs, NetScore_inference
├── zero_shot_eval_results.json                # evaluate --zero_shot (also zero_shot_predictions.json, ...)
└── all_results_<train|evaluate>.json          # sweeps only
```

Key fields in `benchmark_results_<method>.json`:

- **Metrics:** `metrics`, `primary_metric`, `primary_metric_value`, `netscore_performance`; with `--eval_before_training` also `zero_shot_metrics` and `performance_improvement`
- **Parameters:** `total_parameters`, `trainable_parameters`, `trainable_percentage`
- **Training cost:** `training_time_minutes`, `final_training_loss`, `eval_losses_per_epoch`
- **Power and memory:** `avg_gpu_power_watts`, `max_gpu_vram_used_mb`, `peak_torch_allocated_mb`
- **Energy:** `estimated_energy_Wh`, `energy_measurement_source`
- **FLOPs:** `flops` → `seq_len`, `max_new_tokens`, `training_flops_per_sequence`, `inference_flops_per_sequence`, `source`
- **NetScore:** `NetScore`, `netscore_inputs` (training cost); `NetScore_inference`, `netscore_inputs_inference` (inference cost)
- **Inference:** `inference_stats` (without per-sample lists)

---

## Script structure

[lamp_benchmark.py](lamp_benchmark.py) is organized in sections:

| Section | Contents |
|---|---|
| Configuration | `ArchSpec`, `MODELS`, `TASKS`, `METHODS`, `NetScoreConfig` |
| Analytical FLOPs | Transformer / T5 / Mamba-1 / Mamba-2 forward FLOPs, `estimate_flops` |
| NetScore | `calculate_netscore`, `to_netscore_inputs`, `compute_netscore_variants`, `netscore_performance` |
| GPU monitoring | `GPUReader`, `NVMLTrainingCallback`, `GPUInferenceTracker` |
| Mamba2 wrapper | `Mamba2ForCausalLM`, `load_mamba2` |
| Retrieval and prompts | `ContrieverRetriever`, query/corpus makers, PPEP prompt creators, `create_augmented_prompt` |
| Data | `load_lamp_split`, prompt templates, dataset building and tokenization |
| Model loading and methods | `load_tokenizer`, `load_model`, `apply_method`, `LoraPlusTrainer` |
| Evaluation | `generate_predictions`, classification / regression / ROUGE metrics |
| Runners and CLI | `run_training`, `run_evaluation`, argument parsing, sweep planning, summary tables |

## Extending

- **Add a model:** add a `ModelSpec` to `MODELS` (HF id, `seq2seq`/`causal`, LoRA targets, `ArchSpec`). If it needs a chat template, extend `build_generation_prompt` and `build_training_text`. Without an `ArchSpec`, FLOPs fall back to 2 · params · tokens.
- **Add a task:** add a `TaskConfig` to `TASKS` and register a query/corpus maker and a prompt creator in `QUERY_CORPUS_MAKERS` and `PROMPT_CREATORS`. If its metric type is new, extend `compute_task_metrics` and `netscore_performance`.
- **Add a method:** add its name to `METHODS` and `METHOD_NAMES`, implement it in `apply_method`, and add a case to `training_flops` if its training cost differs.
- **Change NetScore:** edit `NetScoreConfig.variants`, the units, or `lamp3_max_error`.

---

## Notes and caveats

- **FLOPs are upper bounds.** They are computed at `max_length` (and `max_new_tokens`), but training batches use dynamic padding and real prompts are often shorter.
- **Evaluation is generation-based.** Per-epoch validation during training reports only the loss, and the final metrics come from the tracked inference pass. It runs one sample at a time, so it can take a while on large validation splits.
- **Retrieval runs on GPU while prompts are built,** then Contriever moves to CPU during training and back to GPU for evaluation. Retrieval cost is not included in training time or energy, but it is included in inference time and energy.
- **Precision:** `--bf16`/`--fp16` set the dtype for loaded weights and for `Trainer` AMP. Without either flag, training loads fp32 weights, and evaluation uses bf16 when the GPU supports it (fp16 otherwise).
- **Mamba-1** runs with `use_cache=False` (as in the original scripts), which makes generation slower.
- **Prompt templates** and the causal-LM training target (the whole sequence) are kept identical to the original scripts, so results stay comparable.
- **`--max_profile_size`** now defaults to 100 for both `train` and `evaluate`. The old `evaluate_lamp.py` used 200; pass `--max_profile_size 200` to reproduce old evaluation numbers.
- **`p` is the trainable-parameter count** in every variant, so `NS` separates PEFT methods sharply and penalizes full fine-tuning heavily.

## License

MIT License
