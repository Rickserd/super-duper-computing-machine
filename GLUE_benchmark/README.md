# GLUE Benchmark Framework

A framework for fine-tuning small LLMs (Transformers and State-Space Models) on GLUE-style classification tasks with several fine-tuning methods, while measuring **GPU energy, power, VRAM, and inference latency**, and scoring runs with the **SAM (Sustainable AI Metric)**.

## Features

| | Supported |
|---|---|
| **Models** | TinyLlama-1.1B, Qwen3-1.7B, Mamba-1 1.4B, Mamba-2 1.3B |
| **Datasets** | SST-2, QNLI, CoLA, STS-B (regression), HellaSwag (4-way multiple choice) |
| **Methods** | BitFit, Full fine-tuning, LoRA, LoRA+, QLoRA |
| **Tracking** | NVML power/VRAM sampling during training, per-sample inference metrics, SAM@1/2/5 |

Every run does the following:

1. Loads the dataset, tokenizer, and model, then applies the method-specific setup.
2. Runs a **zero-shot evaluation** of the untrained classification head.
3. **Fine-tunes** with the Hugging Face `Trainer` while an NVML callback samples GPU power and VRAM.
4. Runs a **post-training evaluation**, keeping the best checkpoint by the dataset's primary metric.
5. Runs an **inference pass** one sample at a time over the eval split (latency, throughput, energy, peak VRAM).
6. Computes SAM from both training energy and inference energy, deletes intermediate checkpoints, saves the model, and writes JSON results.

---

## Requirements

- Linux with an NVIDIA GPU (NVML is used for power readings; bf16 is used when the GPU supports it, fp16 otherwise)
- Python 3.10+ (tested on 3.11)
- PyTorch 2.1+ built for your CUDA version

## Installation

```bash
conda create -n llm python=3.11 -y
conda activate llm

# 1. PyTorch matching your CUDA version (example: CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Core dependencies
pip install -r requirements.txt
```

Optional extras (commented out in `requirements.txt`):

```bash
# Mamba-2 (required for mamba2-1.3b; also speeds up mamba-1.4b)
pip install "causal-conv1d>=1.2.0" mamba-ssm --no-build-isolation

# GPTQ pre-quantized QLoRA for mamba-1.4b
pip install auto-gptq optimum
```

---

## Usage

### Single experiment

```bash
# TinyLlama on SST-2 with LoRA
python main.py --model tinyllama-1.1b --dataset sst2 --method lora

# Qwen3 on CoLA with QLoRA
python main.py --model qwen3-1.7b --dataset cola --method qlora

# HellaSwag prompts (context + 4 endings) are long, so raise --max-length
python main.py --model tinyllama-1.1b --dataset hellaswag --method lora --max-length 256
```

### Sweeps

`--model`, `--dataset`, and `--method` each accept `all`. The runs are the Cartesian product of the three, executed one after another. A failed run is logged and the sweep continues.

```bash
python main.py --model tinyllama-1.1b --dataset sst2 --method all        # all methods
python main.py --model qwen3-1.7b     --dataset all  --method lora       # all datasets
python main.py --model all            --dataset all  --method all --save-summary
```

### Preview and debug

```bash
# List the planned runs without executing them
python main.py --model all --dataset all --method all --dry-run

# Quick smoke test: 1 epoch, eval/save every 50 steps, log every 10
python main.py --model tinyllama-1.1b --dataset sst2 --method lora --debug
```

### CLI reference

| Argument | Default | Description |
|---|---|---|
| `--model, -m` | *required* | `tinyllama-1.1b`, `qwen3-1.7b`, `mamba-1.4b`, `mamba2-1.3b`, `all` |
| `--dataset, -d` | *required* | `sst2`, `qnli`, `cola`, `stsb`, `hellaswag`, `all` |
| `--method, -t` | *required* | `bitfit`, `full_ft`, `lora`, `loraplus`, `qlora`, `all` |
| `--batch-size, -b` | `32` | Per-device train and eval batch size |
| `--epochs, -e` | `5` | Training epochs (forced to 1 with `--debug`) |
| `--learning-rate, -lr` | `1e-5` | Base learning rate |
| `--max-length` | `128` | Max tokenized sequence length |
| `--grad-accum` | `1` | Gradient accumulation steps |
| `--lora-r` | `16` | LoRA rank (LoRA, LoRA+, QLoRA) |
| `--lora-alpha` | `32` | LoRA alpha |
| `--loraplus-ratio` | `16` | LoRA+ learning-rate ratio lr_B / lr_A |
| `--output-dir, -o` | `./outputs` | Base output directory |
| `--gpu-index` | `0` | GPU index used for NVML monitoring |
| `--nvml-no-background-sampling` | off | Disable the background power-sampling thread |
| `--nvml-sample-interval` | `100` | Background sampling interval in ms |
| `--nvml-sample-every-n-steps` | `10` | Also sample every N optimizer steps |
| `--no-save-model` | off | Skip saving weights (results JSON is still written) |
| `--save-summary` | off | Write `benchmark_summary_<timestamp>.json` for the whole sweep |
| `--dry-run` | off | Print planned runs and exit |
| `--quiet, -q` | off | Suppress per-run verbose logging |
| `--debug` | off | 1 epoch, eval/save every 50 steps, log every 10 steps |

Other defaults live in `TrainingConfig` in [config.py](config.py): `warmup_ratio=0.1`, `weight_decay=0.01`, `lora_dropout=0.1`, `save_total_limit=1`.

---

## Models

Defined in `MODELS` in [config.py](config.py).

| Key | HF checkpoint | LoRA targets | Notes |
|---|---|---|---|
| `tinyllama-1.1b` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `q,k,v,o_proj` | Native `AutoModelForSequenceClassification` |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | `q,k,v,o_proj` | Native `AutoModelForSequenceClassification` |
| `mamba-1.4b` | `state-spaces/mamba-1.4b-hf` | `x_proj, in_proj` | CausalLM with a custom last-token classification head, loaded in bf16. `dt_proj` is skipped during BnB quantization. |
| `mamba2-1.3b` | `state-spaces/mamba2-1.3b` | `in_proj, x_proj` | Loaded through `mamba_ssm` and uses the GPT-NeoX-20B tokenizer. **Left padding** is required. The `classifier` head is fully trainable under PEFT. |

**Custom heads** are in [utils/custom_models.py](utils/custom_models.py):

- `CausalLMForSequenceClassification` wraps any HF CausalLM. It pools the hidden state of the last non-padding token and passes it through a linear `score` head. It uses MSE loss for regression and cross-entropy for classification.
- `Mamba2ForSequenceClassification` wraps a `mamba_ssm` `MambaLMHeadModel` and pools position `-1`, which is why it needs left padding.

When a custom-head model's configured LoRA target modules are missing, [utils/module_discovery.py](utils/module_discovery.py) finds suitable linear layers automatically, falling back to `"all-linear"`.

## Datasets

Defined in `DATASETS` in [config.py](config.py). Training uses the `train` split. Evaluation uses `validation`, or `test` when there is no validation split.

| Key | Source | Task | Labels | Primary metric |
|---|---|---|---|---|
| `sst2` | `glue/sst2` | Sentiment | 2 | Accuracy |
| `qnli` | `glue/qnli` | Question/sentence NLI (pair) | 2 | Accuracy |
| `cola` | `glue/cola` | Linguistic acceptability | 2 | Matthews correlation |
| `stsb` | `glue/stsb` | Semantic similarity (pair, regression) | 1 | (Pearson + Spearman) / 2 |
| `hellaswag` | `Rowan/hellaswag` | Commonsense completion | 4 | Accuracy |

**Multiple-choice tasks** such as HellaSwag are converted to single-sequence N-way classification by [utils/data_preprocessing.py](utils/data_preprocessing.py). The context and the lettered endings are rendered into one prompt ending in `Answer:`, and the label is the index of the correct ending (`-1` for unlabeled test rows).

## Methods

Implemented in [methods/](methods/). Each method subclasses `BaseTrainer` in [methods/base.py](methods/base.py) and is registered in `TRAINER_REGISTRY` in [methods/\_\_init\_\_.py](methods/__init__.py).

| Method | What is trained | Optimizer | Notes |
|---|---|---|---|
| `bitfit` | Biases, norm weights, classification head | AdamW (Trainer default) | Everything else is frozen |
| `full_ft` | All parameters | `paged_adamw_8bit` if bitsandbytes is installed, else `adamw_torch` | Loaded without `device_map` and moved to a single GPU. Uses 4 dataloader workers. |
| `lora` | LoRA adapters (+ `modules_to_save`) | AdamW (Trainer default) | `r`, `alpha`, and dropout come from the config. `bias="none"`. |
| `loraplus` | LoRA adapters, with lr_B = lr_A × ratio | PEFT `create_loraplus_optimizer` with `Adam8bit` (falls back to AdamW) | Falls back to manual A/B parameter groups if the PEFT helper fails. Linear warmup schedule. |
| `qlora` | LoRA adapters on a 4-bit base model | `PagedAdamW8bit` (falls back to AdamW) | NF4 with double quantization and bf16/fp16 compute. LoRA dropout 0.05. |

**QLoRA with Mamba models:**

- **mamba-1.4b:** If the directory in `quantized_model_path` (`./quantized_models/mamba-1.4b-gptq-4bit`) exists, QLoRA loads that GPTQ checkpoint instead of quantizing on the fly with bitsandbytes. This requires `auto-gptq` and `optimum`. The `qlora.py` docstring refers to a `quantize_mamba.py` script for creating the checkpoint.
- **mamba2-1.3b:** QLoRA is **not supported** unless a pre-quantized checkpoint is available, because `mamba_ssm`'s fused CUDA kernels don't work with bitsandbytes. The run raises `NotImplementedError`, so use LoRA or LoRA+ instead.

---

## Energy and efficiency measurement

### Training ([utils/nvml_callback.py](utils/nvml_callback.py))

`CheckpointNVMLCallback` combines three sources. The first one that has data is used as the energy estimate:

1. **Continuous sampling:** a background thread reads power and VRAM every `--nvml-sample-interval` ms.
2. **Step sampling:** a reading every `--nvml-sample-every-n-steps` optimizer steps.
3. **Checkpoint sampling:** a reading at each checkpoint save. This is kept only as a fallback because it mostly captures I/O rather than compute.

Energy is integrated with the trapezoidal rule, `E = ∫ P dt`, and reported in Wh. The callback also records PyTorch allocator peaks between checkpoints.

### Inference ([utils/inference_tracker.py](utils/inference_tracker.py))

After training, the optimizer and scheduler are freed. The eval split is then run **one sample at a time**, with each sample padded to `max_length`. `GPUInferenceTracker` polls power and VRAM every 50 ms and records:

- total time and forward-only time
- throughput
- energy per sample and in total
- average power
- peak VRAM

It falls back to `nvidia-smi` when `pynvml` is unavailable.

### SAM: Sustainable AI Metric ([utils/metrics.py](utils/metrics.py))

```
SAM@α = score^α × α / log(Wh)        (natural log; log(Wh + 1) is used when Wh ≤ 1)
```

- `score` is the dataset's primary metric (accuracy, MCC, or combined Pearson/Spearman).
- `α ∈ {1, 2, 5}`.
- SAM is computed twice: from **training** energy (`SAM@1/2/5`) and from **inference** energy (`SAM_inference`).
- It is `null` when no energy data is available.

---

## Output structure

```
outputs/
├── benchmark_summary_<timestamp>.json         # with --save-summary
└── <model>/
    └── <dataset>/
        └── <method>_<YYYYmmdd_HHMMSS>/
            ├── benchmark_results_<method>.json # metrics, params, energy, SAM, inference stats
            ├── power_vram_timeseries.json      # step/checkpoint samples + summary
            ├── inference_stats.json            # aggregate + per-sample inference metrics
            └── model / adapter + tokenizer files (unless --no-save-model)
```

Key fields in `benchmark_results_<method>.json`:

- **Metrics:** `zero_shot_<metric>`, `fine_tuned_<metric>`, `improvement`
- **Parameters:** `total_parameters`, `trainable_parameters`, `trainable_percentage`
- **Training cost:** `training_time_minutes`, `final_training_loss`
- **Power and memory:** `avg_gpu_power_watts`, `max_gpu_vram_used_mb`
- **Energy:** `estimated_energy_Wh`, `energy_measurement_source`
- **SAM:** `SAM@1`, `SAM@2`, `SAM@5`, `SAM_inference`
- **Inference:** `inference_stats`

Custom-head and Mamba-2 models are saved as `.bin` rather than safetensors, because their wrapper classes contain shared tensors.

---

## Project structure

```
.
├── main.py                    # CLI entry point; builds the run matrix and prints the results table
├── config.py                  # ModelConfig / DatasetConfig / TrainingConfig registries
├── methods/
│   ├── __init__.py            # TRAINER_REGISTRY + get_trainer()
│   ├── base.py                # BaseTrainer: data, tokenization, Trainer, eval, inference pass, results
│   ├── bitfit.py
│   ├── full_ft.py
│   ├── lora.py
│   ├── loraplus.py
│   └── qlora.py
└── utils/
    ├── custom_models.py       # CausalLM and Mamba2 sequence-classification wrappers
    ├── data_preprocessing.py  # Multiple-choice → single-sequence classification
    ├── inference_tracker.py   # GPUInferenceTracker
    ├── metrics.py             # SAM calculation + metric helpers
    ├── module_discovery.py    # LoRA target-module auto-discovery
    ├── nvml_callback.py       # Training-time NVML power/VRAM callback
    └── seed.py                # set_seed() helper
```

---

## Extending the framework

### Add a model

```python
# config.py
MODELS["new-model"] = ModelConfig(
    name="new-model",
    hf_name="organization/model-name",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    description="Description of the model",
    # requires_custom_head=True,     # if there is no native SequenceClassification support
    # padding_side="left",
    # torch_dtype=torch.bfloat16,
)
```

Also add the key to the `--model` examples above.

### Add a dataset

```python
# config.py
DATASETS["new-dataset"] = DatasetConfig(
    name="new-dataset",
    hf_name="glue",                 # or any HF dataset id
    subset="new-subset",            # or None
    text_column="sentence",
    text_column_2=None,             # second column for sentence-pair tasks
    label_column="label",
    num_labels=2,
    id2label={0: "neg", 1: "pos"},
    label2id={"neg": 0, "pos": 1},
    metric_name="accuracy",         # "accuracy" | "matthews_correlation" | "pearson_spearman"
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    # is_regression=True,           # for num_labels=1 regression
    # is_multiple_choice=True, context_column="ctx", choices_column="endings",
)
```

### Add a method

1. Create `methods/my_method.py` with a `BaseTrainer` subclass. Set `method_name` and implement `_apply_method_specific_setup()`. Override `_load_model`, `_get_training_args`, or `_setup_trainer` if the method needs to.
2. Register the class in `TRAINER_REGISTRY` in `methods/__init__.py`.
3. Add its name to `METHODS` in `config.py`.

---

## Notes and caveats

- **Experiment tracking:** `report_to=None` in `TrainingArguments` means "all installed integrations" on transformers 4.x. If `wandb` is installed, runs are logged to it. Set `WANDB_MODE=disabled` or uninstall it to opt out.
- **Seeding:** `utils/seed.py` provides `set_seed()`, but `main.py` does not call it. Runs are not seeded beyond the `Trainer` default (seed 42).
- **`--max-length` default:** the CLI default is **128**, which overrides `TrainingConfig.max_length = 256`.
- **`--nvml-background-sampling`:** this flag has no effect, since background sampling is on by default. Use `--nvml-no-background-sampling` to turn it off.
- **LoRA dropout:** it is not exposed on the CLI. Edit `TrainingConfig.lora_dropout` to change it. QLoRA always uses 0.05.
- **Inference pass length:** the pass iterates over the full eval split one sample at a time, so it can take a while on larger splits such as QNLI (~5.5k) or HellaSwag (~10k).

## License

MIT License
