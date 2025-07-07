"""
Sales Conversion AI Pipeline Package
Provides embedder, scorer, and LangChain integration for sales conversion prediction
"""

from .embedder import SalesEmbedder, EmbeddingComparison
from .scorer import (
    SimilarityScorer, 
    CosineSimilarityScorer, 
    EuclideanDistanceScorer, 
    MMRScorer, 
    HybridScorer,
    ConversionPredictor,
    ThresholdOptimizer
)
from .langchain_chain import SalesConversionChain, SalesConversionRAGChain

__version__ = "1.0.0"
__author__ = "Sales Conversion AI Team"

__all__ = [
    # Embedder
    "SalesEmbedder",
    "EmbeddingComparison",
    
    # Scorers
    "SimilarityScorer",
    "CosineSimilarityScorer", 
    "EuclideanDistanceScorer",
    "MMRScorer",
    "HybridScorer",
    "ConversionPredictor",
    "ThresholdOptimizer",
    
    # LangChain Integration
    "SalesConversionChain",
    "SalesConversionRAGChain"
] 