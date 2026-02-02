# Methods package
from .base import BaseTrainer
from .bitfit import BitFitTrainer
from .full_ft import FullFTTrainer
from .lora import LoRATrainer
from .loraplus import LoraPlusTrainer
from .qlora import QLoRATrainer

__all__ = [
    "BaseTrainer",
    "BitFitTrainer", 
    "FullFTTrainer",
    "LoRATrainer",
    "LoraPlusTrainer",
    "QLoRATrainer",
]

# Method name to trainer class mapping
TRAINER_REGISTRY = {
    "bitfit": BitFitTrainer,
    "full_ft": FullFTTrainer,
    "lora": LoRATrainer,
    "loraplus": LoraPlusTrainer,
    "qlora": QLoRATrainer,
}


def get_trainer(method_name: str):
    """Get trainer class by method name."""
    if method_name not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[method_name]

