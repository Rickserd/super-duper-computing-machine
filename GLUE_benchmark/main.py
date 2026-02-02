#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GLUE Benchmark Framework - Main Entry Point

Unified script for fine-tuning LLMs on GLUE benchmark tasks using various
parameter-efficient fine-tuning (PEFT) methods.

Features:
- Multiple models: TinyLlama-1.1B, Qwen3-1.7B
- Multiple datasets: SST-2, QNLI, CoLA
- Multiple methods: BitFit, Full FT, LoRA, LoRA+, QLoRA
- Energy efficiency tracking via NVML
- SAM metric calculation

Usage:
    # Single run
    python main.py --model tinyllama-1.1b --dataset sst2 --method lora
    
    # Run all methods on one dataset
    python main.py --model qwen3-1.7b --dataset sst2 --method all
    
    # Run all datasets with one method
    python main.py --model tinyllama-1.1b --dataset all --method lora
    
    # Run everything
    python main.py --model tinyllama-1.1b --dataset all --method all
    
    # Debug mode with fewer epochs
    python main.py --model tinyllama-1.1b --dataset sst2 --method lora --debug
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

from config import (
    MODELS, DATASETS, METHODS,
    get_model_config, get_dataset_config,
    TrainingConfig,
)
from methods import get_trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GLUE Benchmark Framework for LLM Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model tinyllama-1.1b --dataset sst2 --method lora
  python main.py --model qwen3-1.7b --dataset all --method all
  python main.py --model tinyllama-1.1b --dataset cola --method bitfit --debug
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=list(MODELS.keys()) + ["all"],
        help=f"Model to use. Available: {', '.join(MODELS.keys())}, all"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=list(DATASETS.keys()) + ["all"],
        help=f"Dataset to use. Available: {', '.join(DATASETS.keys())}, all"
    )
    
    parser.add_argument(
        "--method", "-t",
        type=str,
        required=True,
        choices=METHODS + ["all"],
        help=f"Training method. Available: {', '.join(METHODS)}, all"
    )
    
    # Training configuration
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable fully deterministic mode (slower but exactly reproducible)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=5,
        help="Number of epochs (default: 5)"
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length (default: 128)"
    )
    
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    
    # LoRA specific
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    
    parser.add_argument(
        "--loraplus-ratio",
        type=int,
        default=16,
        help="LoRA+ learning rate ratio (default: 16)"
    )
    
    # Output and logging
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./outputs",
        help="Base output directory (default: ./outputs)"
    )
    
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index for NVML monitoring (default: 0)"
    )
    
    # NVML power sampling configuration
    parser.add_argument(
        "--nvml-background-sampling",
        action="store_true",
        default=True,
        help="Use continuous background sampling for power (default: True)"
    )
    
    parser.add_argument(
        "--nvml-no-background-sampling",
        action="store_true",
        help="Disable background sampling (only sample at training steps)"
    )
    
    parser.add_argument(
        "--nvml-sample-interval",
        type=int,
        default=100,
        help="Background sampling interval in ms (default: 100ms = 10 samples/sec)"
    )
    
    parser.add_argument(
        "--nvml-sample-every-n-steps",
        type=int,
        default=10,
        help="Sample power every N training steps (default: 10)"
    )
    
    # Debugging and testing
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: 1 epoch, reduced eval steps"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually running"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Save a summary JSON of all runs"
    )
    
    return parser.parse_args()


def get_run_configs(args) -> List[Dict[str, str]]:
    """Generate list of run configurations based on arguments."""
    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    methods = METHODS if args.method == "all" else [args.method]
    
    configs = []
    for model in models:
        for dataset in datasets:
            for method in methods:
                configs.append({
                    "model": model,
                    "dataset": dataset,
                    "method": method,
                })
    
    return configs


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    method_name: str,
    training_config: TrainingConfig,
    base_output_dir: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single fine-tuning experiment."""
    
    # Get configurations
    model_config = get_model_config(model_name)
    dataset_config = get_dataset_config(dataset_name)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        base_output_dir,
        model_name,
        dataset_name,
        f"{method_name}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Get trainer class and instantiate
    trainer_cls = get_trainer(method_name)
    trainer = trainer_cls(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        output_dir=output_dir,
        verbose=verbose,
    )
    
    # Run training
    results = trainer.run()
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set random seed for reproducibility
    from utils.seed import set_seed
    set_seed(args.seed, deterministic=args.deterministic)
    
    # Build training config
    use_background_sampling = not args.nvml_no_background_sampling
    
    training_config = TrainingConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=1 if args.debug else args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_accumulation_steps=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        loraplus_ratio=args.loraplus_ratio,
        gpu_index=args.gpu_index,
        eval_steps=50 if args.debug else 200,
        save_steps=50 if args.debug else 200,
        logging_steps=10 if args.debug else 50,
        # NVML power sampling configuration
        nvml_use_background_sampling=use_background_sampling,
        nvml_sample_interval_ms=args.nvml_sample_interval,
        nvml_sample_every_n_steps=args.nvml_sample_every_n_steps,
    )
    
    # Get all run configurations
    run_configs = get_run_configs(args)
    
    print("="*70)
    print("🚀 GLUE Benchmark Framework")
    print("="*70)
    print(f"📋 Planned experiments: {len(run_configs)}")
    for i, cfg in enumerate(run_configs, 1):
        print(f"   {i}. {cfg['model']} + {cfg['dataset']} + {cfg['method']}")
    print("="*70)
    
    if args.dry_run:
        print("\n⚠️  Dry run mode - no experiments will be executed.")
        return
    
    # Run experiments
    all_results = []
    successful = 0
    failed = 0
    
    for i, cfg in enumerate(run_configs, 1):
        print(f"\n{'='*70}")
        print(f"📌 Experiment {i}/{len(run_configs)}")
        print(f"   Model: {cfg['model']}")
        print(f"   Dataset: {cfg['dataset']}")
        print(f"   Method: {cfg['method']}")
        print("="*70)
        
        try:
            results = run_single_experiment(
                model_name=cfg["model"],
                dataset_name=cfg["dataset"],
                method_name=cfg["method"],
                training_config=training_config,
                base_output_dir=args.output_dir,
                verbose=not args.quiet,
            )
            results["status"] = "success"
            all_results.append(results)
            successful += 1
            
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            
            all_results.append({
                "model": cfg["model"],
                "dataset": cfg["dataset"],
                "method": cfg["method"],
                "status": "failed",
                "error": str(e),
            })
            failed += 1
    
    # Print final summary
    print("\n" + "="*70)
    print("🏁 BENCHMARK COMPLETE")
    print("="*70)
    print(f"✅ Successful: {successful}/{len(run_configs)}")
    print(f"❌ Failed: {failed}/{len(run_configs)}")
    
    # Print results table
    if successful > 0:
        print("\n📊 Results Summary:")
        print("-"*100)
        header = f"{'Model':<20} {'Dataset':<10} {'Method':<12} {'Metric':<10} {'Score':<10} {'Energy(Wh)':<12} {'SAM@1':<12}"
        print(header)
        print("-"*100)
        
        for r in all_results:
            if r["status"] == "success":
                model = r.get("model_name", "N/A")[:18]
                dataset = r.get("dataset", "N/A")
                method = r.get("method", "N/A")
                metric = r.get("metric_name", "acc")
                score = r.get(f"fine_tuned_{metric}", r.get("fine_tuned_accuracy", 0))
                energy = r.get("estimated_energy_Wh")
                sam1 = r.get("SAM@1")
                
                energy_str = f"{energy:.4f}" if energy else "N/A"
                sam1_str = f"{sam1:.6f}" if sam1 else "N/A"
                
                print(f"{model:<20} {dataset:<10} {method:<12} {metric:<10} {score:<10.4f} {energy_str:<12} {sam1_str:<12}")
        
        print("-"*100)
    
    # Save summary
    if args.save_summary:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(args.output_dir, f"benchmark_summary_{timestamp}.json")
        os.makedirs(args.output_dir, exist_ok=True)
        
        summary = {
            "timestamp": timestamp,
            "args": vars(args),
            "num_experiments": len(run_configs),
            "successful": successful,
            "failed": failed,
            "results": all_results,
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

