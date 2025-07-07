"""
Sales Conversion AI Training Package
Provides contrastive dataset creation, model training, and evaluation
"""

from .contrastive_dataset import ContrastiveDatasetCreator, ContrastivePair
from .train_finetune import (
    ContrastiveDataset,
    TripletLoss,
    ContrastiveModel,
    SalesEmbeddingTrainer
)
from .evaluation import SalesConversionEvaluator

__version__ = "1.0.0"
__author__ = "Sales Conversion AI Team"

__all__ = [
    # Dataset Creation
    "ContrastiveDatasetCreator",
    "ContrastivePair",
    
    # Training
    "ContrastiveDataset",
    "TripletLoss", 
    "ContrastiveModel",
    "SalesEmbeddingTrainer",
    
    # Evaluation
    "SalesConversionEvaluator"
] 