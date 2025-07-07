"""
Scorer Module for Sales Conversion Prediction
Implements similarity scoring and conversion prediction logic
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversionPrediction:
    """Represents a conversion prediction result"""
    conversion_score: float
    conversion_probability: float
    prediction: int
    confidence: float
    similar_cases: List[Dict[str, Any]]
    reasoning: str

class SimilarityScorer:
    """Base class for similarity scoring methods"""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = embeddings
        self.labels = labels
        self.converted_indices = np.where(labels == 1)[0]
        self.non_converted_indices = np.where(labels == 0)[0]
        
    def compute_similarity(self, query_embedding: np.ndarray, 
                          reference_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity between query and reference embeddings"""
        raise NotImplementedError
    
    def score_conversion(self, query_embedding: np.ndarray) -> float:
        """Score conversion likelihood based on similarity"""
        raise NotImplementedError

class CosineSimilarityScorer(SimilarityScorer):
    """Cosine similarity-based scorer"""
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          reference_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity"""
        return cosine_similarity([query_embedding], reference_embeddings)[0]
    
    def score_conversion(self, query_embedding: np.ndarray) -> float:
        """Score based on cosine similarity to converted vs non-converted prototypes"""
        # Compute similarities to converted and non-converted examples
        converted_similarities = self.compute_similarity(
            query_embedding, 
            self.embeddings[self.converted_indices]
        )
        non_converted_similarities = self.compute_similarity(
            query_embedding, 
            self.embeddings[self.non_converted_indices]
        )
        
        # Return difference in mean similarities
        return np.mean(converted_similarities) - np.mean(non_converted_similarities)

class EuclideanDistanceScorer(SimilarityScorer):
    """Euclidean distance-based scorer"""
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          reference_embeddings: np.ndarray) -> np.ndarray:
        """Compute negative euclidean distance (higher = more similar)"""
        distances = euclidean_distances([query_embedding], reference_embeddings)[0]
        return -distances  # Convert to similarity
    
    def score_conversion(self, query_embedding: np.ndarray) -> float:
        """Score based on euclidean distance to converted vs non-converted prototypes"""
        # Compute distances to converted and non-converted examples
        converted_distances = euclidean_distances(
            [query_embedding], 
            self.embeddings[self.converted_indices]
        )[0]
        non_converted_distances = euclidean_distances(
            [query_embedding], 
            self.embeddings[self.non_converted_indices]
        )[0]
        
        # Return difference in mean distances (negative because lower distance = more similar)
        return np.mean(non_converted_distances) - np.mean(converted_distances)

class MMRScorer(SimilarityScorer):
    """Maximum Marginal Relevance scorer for diverse similarity"""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, lambda_param: float = 0.5):
        super().__init__(embeddings, labels)
        self.lambda_param = lambda_param  # Balance between relevance and diversity
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          reference_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity for MMR"""
        return cosine_similarity([query_embedding], reference_embeddings)[0]
    
    def score_conversion(self, query_embedding: np.ndarray) -> float:
        """Score using MMR approach"""
        # Get top similar converted and non-converted examples
        converted_similarities = self.compute_similarity(
            query_embedding, 
            self.embeddings[self.converted_indices]
        )
        non_converted_similarities = self.compute_similarity(
            query_embedding, 
            self.embeddings[self.non_converted_indices]
        )
        
        # Get top-k most similar examples from each class
        k = min(5, len(converted_similarities), len(non_converted_similarities))
        
        top_converted_indices = np.argsort(converted_similarities)[-k:]
        top_non_converted_indices = np.argsort(non_converted_similarities)[-k:]
        
        # Compute MMR scores
        converted_mmr = np.mean(converted_similarities[top_converted_indices])
        non_converted_mmr = np.mean(non_converted_similarities[top_non_converted_indices])
        
        return converted_mmr - non_converted_mmr

class HybridScorer(SimilarityScorer):
    """Hybrid scorer combining multiple similarity methods"""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, 
                 weights: Dict[str, float] = None):
        super().__init__(embeddings, labels)
        self.weights = weights or {
            'cosine': 0.4,
            'euclidean': 0.3,
            'mmr': 0.3
        }
        
        # Initialize individual scorers
        self.cosine_scorer = CosineSimilarityScorer(embeddings, labels)
        self.euclidean_scorer = EuclideanDistanceScorer(embeddings, labels)
        self.mmr_scorer = MMRScorer(embeddings, labels)
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          reference_embeddings: np.ndarray) -> np.ndarray:
        """Compute weighted combination of similarities"""
        cosine_sim = self.cosine_scorer.compute_similarity(query_embedding, reference_embeddings)
        euclidean_sim = self.euclidean_scorer.compute_similarity(query_embedding, reference_embeddings)
        mmr_sim = self.mmr_scorer.compute_similarity(query_embedding, reference_embeddings)
        
        # Normalize euclidean similarities to [0, 1] range
        euclidean_sim = (euclidean_sim - np.min(euclidean_sim)) / (np.max(euclidean_sim) - np.min(euclidean_sim))
        
        # Weighted combination
        combined_sim = (
            self.weights['cosine'] * cosine_sim +
            self.weights['euclidean'] * euclidean_sim +
            self.weights['mmr'] * mmr_sim
        )
        
        return combined_sim
    
    def score_conversion(self, query_embedding: np.ndarray) -> float:
        """Score using weighted combination of methods"""
        cosine_score = self.cosine_scorer.score_conversion(query_embedding)
        euclidean_score = self.euclidean_scorer.score_conversion(query_embedding)
        mmr_score = self.mmr_scorer.score_conversion(query_embedding)
        
        # Normalize scores to similar ranges
        scores = [cosine_score, euclidean_score, mmr_score]
        normalized_scores = []
        
        for score in scores:
            if np.std(scores) > 0:
                normalized_score = (score - np.mean(scores)) / np.std(scores)
            else:
                normalized_score = 0
            normalized_scores.append(normalized_score)
        
        # Weighted combination
        final_score = (
            self.weights['cosine'] * normalized_scores[0] +
            self.weights['euclidean'] * normalized_scores[1] +
            self.weights['mmr'] * normalized_scores[2]
        )
        
        return final_score

class ConversionPredictor:
    """Main class for conversion prediction using similarity scoring"""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, 
                 transcripts: List[str], scorer_type: str = 'hybrid'):
        self.embeddings = embeddings
        self.labels = labels
        self.transcripts = transcripts
        
        # Initialize scorer
        if scorer_type == 'cosine':
            self.scorer = CosineSimilarityScorer(embeddings, labels)
        elif scorer_type == 'euclidean':
            self.scorer = EuclideanDistanceScorer(embeddings, labels)
        elif scorer_type == 'mmr':
            self.scorer = MMRScorer(embeddings, labels)
        else:
            self.scorer = HybridScorer(embeddings, labels)
        
        # Compute prototypes for reference
        self._compute_prototypes()
    
    def _compute_prototypes(self):
        """Compute prototype embeddings for each class"""
        converted_indices = np.where(self.labels == 1)[0]
        non_converted_indices = np.where(self.labels == 0)[0]
        
        self.converted_prototype = np.mean(self.embeddings[converted_indices], axis=0)
        self.non_converted_prototype = np.mean(self.embeddings[non_converted_indices], axis=0)
    
    def predict_conversion(self, query_embedding: np.ndarray, 
                          top_k_similar: int = 5) -> ConversionPrediction:
        """Predict conversion likelihood for a query embedding"""
        
        # Compute conversion score
        conversion_score = self.scorer.score_conversion(query_embedding)
        
        # Convert score to probability using sigmoid
        conversion_probability = 1 / (1 + np.exp(-conversion_score * 3))  # Scale factor of 3
        
        # Make prediction
        prediction = int(conversion_probability > 0.5)
        
        # Compute confidence based on distance to decision boundary
        confidence = abs(conversion_probability - 0.5) * 2  # Scale to [0, 1]
        
        # Find similar cases
        similar_cases = self._find_similar_cases(query_embedding, top_k_similar)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(conversion_score, similar_cases)
        
        return ConversionPrediction(
            conversion_score=float(conversion_score),
            conversion_probability=float(conversion_probability),
            prediction=prediction,
            confidence=float(confidence),
            similar_cases=similar_cases,
            reasoning=reasoning
        )
    
    def _find_similar_cases(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Find most similar cases to the query"""
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_cases = []
        for idx in top_indices:
            similar_cases.append({
                'transcript': self.transcripts[idx],
                'similarity': float(similarities[idx]),
                'actual_conversion': int(self.labels[idx]),
                'rank': len(similar_cases) + 1
            })
        
        return similar_cases
    
    def _generate_reasoning(self, conversion_score: float, 
                           similar_cases: List[Dict[str, Any]]) -> str:
        """Generate reasoning for the prediction"""
        if conversion_score > 0.1:
            sentiment = "positive"
            direction = "likely to convert"
        elif conversion_score < -0.1:
            sentiment = "negative"
            direction = "unlikely to convert"
        else:
            sentiment = "neutral"
            direction = "uncertain"
        
        # Count similar cases by conversion status
        converted_similar = sum(1 for case in similar_cases if case['actual_conversion'] == 1)
        total_similar = len(similar_cases)
        
        reasoning = f"This call shows {sentiment} conversion signals. "
        reasoning += f"Based on {total_similar} similar historical cases, "
        reasoning += f"{converted_similar} resulted in conversions. "
        reasoning += f"The call is {direction}."
        
        return reasoning
    
    def predict_batch(self, query_embeddings: np.ndarray) -> List[ConversionPrediction]:
        """Predict conversion for multiple queries"""
        predictions = []
        for embedding in query_embeddings:
            prediction = self.predict_conversion(embedding)
            predictions.append(prediction)
        
        return predictions
    
    def evaluate_predictions(self, test_embeddings: np.ndarray, 
                           test_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate prediction performance on test data"""
        predictions = self.predict_batch(test_embeddings)
        
        # Extract predictions and probabilities
        pred_labels = [p.prediction for p in predictions]
        pred_probabilities = [p.conversion_probability for p in predictions]
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(test_labels, pred_labels),
            'precision': precision_score(test_labels, pred_labels, zero_division=0),
            'recall': recall_score(test_labels, pred_labels, zero_division=0),
            'f1_score': f1_score(test_labels, pred_labels, zero_division=0),
            'auc_roc': roc_auc_score(test_labels, pred_probabilities),
            'mean_confidence': np.mean([p.confidence for p in predictions])
        }
        
        return metrics

class ThresholdOptimizer:
    """Optimize prediction thresholds for better performance"""
    
    def __init__(self, predictor: ConversionPredictor):
        self.predictor = predictor
    
    def optimize_threshold(self, val_embeddings: np.ndarray, val_labels: np.ndarray,
                          metric: str = 'f1_score') -> float:
        """Find optimal threshold for the specified metric"""
        predictions = self.predictor.predict_batch(val_embeddings)
        scores = [p.conversion_score for p in predictions]
        
        # Try different thresholds
        thresholds = np.linspace(-2, 2, 100)
        best_threshold = 0
        best_score = 0
        
        for threshold in thresholds:
            # Apply threshold
            pred_labels = [1 if score > threshold else 0 for score in scores]
            
            # Compute metric
            if metric == 'accuracy':
                score = accuracy_score(val_labels, pred_labels)
            elif metric == 'precision':
                score = precision_score(val_labels, pred_labels, zero_division=0)
            elif metric == 'recall':
                score = recall_score(val_labels, pred_labels, zero_division=0)
            elif metric == 'f1_score':
                score = f1_score(val_labels, pred_labels, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold

def main():
    """Example usage of the scorer"""
    # Load sample data
    import pandas as pd
    from pipeline.embedder import SalesEmbedder
    
    # Load data
    df = pd.read_csv('data/sample_transcripts.csv')
    transcripts = df['call_transcript'].tolist()
    labels = df['conversion_label'].values
    
    # Load embedder and compute embeddings
    embedder = SalesEmbedder(model_path='models/finetuned_sales_model')
    embeddings = embedder.encode(transcripts)
    
    # Initialize predictor
    predictor = ConversionPredictor(embeddings, labels, transcripts, scorer_type='hybrid')
    
    # Test prediction
    test_transcript = "Agent: Hi, I'm calling about your recent inquiry. Customer: Yes, we're interested in your solution. Agent: Great! Let me tell you about our features..."
    test_embedding = embedder.encode(test_transcript)
    
    prediction = predictor.predict_conversion(test_embedding)
    print("Prediction:", prediction)

if __name__ == "__main__":
    main() 