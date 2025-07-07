"""
Embedder Module for Sales Conversion Prediction
Loads and applies fine-tuned embedding models
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Union
import logging
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesEmbedder:
    """Embedder for sales call transcripts using fine-tuned models"""
    
    def __init__(self, model_path: str = None, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.converted_prototype = None
        self.non_converted_prototype = None
        self.is_finetuned = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading fine-tuned model from {self.model_path}")
                self.model = SentenceTransformer(self.model_path)
                self.is_finetuned = True
            else:
                logger.info(f"Loading generic model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.is_finetuned = False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def compute_prototypes(self, training_transcripts: List[str], training_labels: List[int]):
        """Compute prototype embeddings for converted and non-converted classes"""
        logger.info("Computing prototype embeddings...")
        
        # Encode all training transcripts
        embeddings = self.encode(training_transcripts)
        
        # Separate by class
        converted_embeddings = embeddings[np.array(training_labels) == 1]
        non_converted_embeddings = embeddings[np.array(training_labels) == 0]
        
        if len(converted_embeddings) == 0 or len(non_converted_embeddings) == 0:
            logger.warning("No examples in one of the classes")
            return
        
        # Compute prototypes (mean embeddings)
        self.converted_prototype = np.mean(converted_embeddings, axis=0)
        self.non_converted_prototype = np.mean(non_converted_embeddings, axis=0)
        
        logger.info(f"Computed prototypes - Converted: {len(converted_embeddings)} samples, "
                   f"Non-converted: {len(non_converted_embeddings)} samples")
    
    def predict_conversion_score(self, transcript: str) -> Dict[str, float]:
        """Predict conversion likelihood for a single transcript"""
        if self.converted_prototype is None or self.non_converted_prototype is None:
            raise ValueError("Prototypes not computed. Call compute_prototypes() first.")
        
        # Encode transcript
        embedding = self.encode(transcript)
        
        # Compute similarities to prototypes
        converted_similarity = cosine_similarity([embedding], [self.converted_prototype])[0][0]
        non_converted_similarity = cosine_similarity([embedding], [self.non_converted_prototype])[0][0]
        
        # Compute conversion score (difference in similarities)
        conversion_score = converted_similarity - non_converted_similarity
        
        # Convert to probability (sigmoid-like transformation)
        probability = 1 / (1 + np.exp(-conversion_score * 5))  # Scale factor of 5
        
        return {
            'conversion_score': float(conversion_score),
            'conversion_probability': float(probability),
            'converted_similarity': float(converted_similarity),
            'non_converted_similarity': float(non_converted_similarity),
            'prediction': int(probability > 0.5)
        }
    
    def predict_batch(self, transcripts: List[str]) -> List[Dict[str, float]]:
        """Predict conversion likelihood for multiple transcripts"""
        if self.converted_prototype is None or self.non_converted_prototype is None:
            raise ValueError("Prototypes not computed. Call compute_prototypes() first.")
        
        # Encode all transcripts
        embeddings = self.encode(transcripts)
        
        # Compute similarities to prototypes
        converted_similarities = cosine_similarity(embeddings, [self.converted_prototype]).flatten()
        non_converted_similarities = cosine_similarity(embeddings, [self.non_converted_prototype]).flatten()
        
        # Compute conversion scores
        conversion_scores = converted_similarities - non_converted_similarities
        probabilities = 1 / (1 + np.exp(-conversion_scores * 5))
        
        # Create results
        results = []
        for i in range(len(transcripts)):
            results.append({
                'conversion_score': float(conversion_scores[i]),
                'conversion_probability': float(probabilities[i]),
                'converted_similarity': float(converted_similarities[i]),
                'non_converted_similarity': float(non_converted_similarities[i]),
                'prediction': int(probabilities[i] > 0.5)
            })
        
        return results
    
    def find_similar_transcripts(self, query_transcript: str, 
                               candidate_transcripts: List[str], 
                               top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar transcripts to the query"""
        # Encode query and candidates
        query_embedding = self.encode(query_transcript)
        candidate_embeddings = self.encode(candidate_transcripts)
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create results
        results = []
        for idx in top_indices:
            results.append({
                'transcript': candidate_transcripts[idx],
                'similarity': float(similarities[idx]),
                'rank': len(results) + 1
            })
        
        return results
    
    def save_prototypes(self, output_path: str):
        """Save computed prototypes to file"""
        if self.converted_prototype is None or self.non_converted_prototype is None:
            raise ValueError("Prototypes not computed yet")
        
        data = {
            'converted_prototype': self.converted_prototype.tolist(),
            'non_converted_prototype': self.non_converted_prototype.tolist(),
            'model_path': self.model_path,
            'model_name': self.model_name,
            'is_finetuned': self.is_finetuned
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Prototypes saved to {output_path}")
    
    def load_prototypes(self, input_path: str):
        """Load computed prototypes from file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.converted_prototype = np.array(data['converted_prototype'])
        self.non_converted_prototype = np.array(data['non_converted_prototype'])
        self.model_path = data.get('model_path')
        self.model_name = data.get('model_name')
        self.is_finetuned = data.get('is_finetuned', False)
        
        logger.info(f"Prototypes loaded from {input_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'model_name': self.model_name,
            'is_finetuned': self.is_finetuned,
            'prototypes_computed': self.converted_prototype is not None,
            'embedding_dimension': self.model.get_sentence_embedding_dimension() if self.model else None
        }

class EmbeddingComparison:
    """Compare embeddings from different models"""
    
    def __init__(self, generic_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.generic_model_name = generic_model_name
        self.generic_embedder = None
        self.finetuned_embedder = None
    
    def load_models(self, finetuned_model_path: str):
        """Load both generic and fine-tuned models"""
        self.generic_embedder = SalesEmbedder(model_name=self.generic_model_name)
        self.finetuned_embedder = SalesEmbedder(model_path=finetuned_model_path)
        
        logger.info("Both models loaded successfully")
    
    def compare_embeddings(self, transcripts: List[str]) -> Dict[str, np.ndarray]:
        """Compare embeddings from both models"""
        logger.info("Computing embeddings with both models...")
        
        generic_embeddings = self.generic_embedder.encode(transcripts)
        finetuned_embeddings = self.finetuned_embedder.encode(transcripts)
        
        return {
            'generic': generic_embeddings,
            'finetuned': finetuned_embeddings
        }
    
    def compute_similarity_differences(self, transcripts: List[str]) -> List[float]:
        """Compute how much embeddings differ between models"""
        embeddings = self.compare_embeddings(transcripts)
        
        differences = []
        for i in range(len(transcripts)):
            # Compute cosine similarity between generic and fine-tuned embeddings
            similarity = cosine_similarity(
                [embeddings['generic'][i]], 
                [embeddings['finetuned'][i]]
            )[0][0]
            differences.append(1 - similarity)  # Convert to difference
        
        return differences
    
    def analyze_embedding_changes(self, transcripts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Analyze how fine-tuning changed embeddings"""
        embeddings = self.compare_embeddings(transcripts)
        differences = self.compute_similarity_differences(transcripts)
        
        # Analyze differences by conversion status
        converted_diffs = [d for d, l in zip(differences, labels) if l == 1]
        non_converted_diffs = [d for d, l in zip(differences, labels) if l == 0]
        
        analysis = {
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            'converted_mean_difference': np.mean(converted_diffs) if converted_diffs else 0,
            'non_converted_mean_difference': np.mean(non_converted_diffs) if non_converted_diffs else 0,
            'max_difference': np.max(differences),
            'min_difference': np.min(differences)
        }
        
        return analysis

def main():
    """Example usage of the embedder"""
    # Initialize embedder
    embedder = SalesEmbedder(model_path='models/finetuned_sales_model')
    
    # Load training data to compute prototypes
    import pandas as pd
    df = pd.read_csv('data/sample_transcripts.csv')
    transcripts = df['call_transcript'].tolist()
    labels = df['conversion_label'].tolist()
    
    # Compute prototypes
    embedder.compute_prototypes(transcripts, labels)
    
    # Test prediction
    test_transcript = "Agent: Hi, I'm calling about your recent inquiry. Customer: Yes, we're interested in your solution. Agent: Great! Let me tell you about our features..."
    
    prediction = embedder.predict_conversion_score(test_transcript)
    print("Prediction:", prediction)
    
    # Save prototypes
    embedder.save_prototypes('models/prototypes.json')

if __name__ == "__main__":
    main() 