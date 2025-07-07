"""
Contrastive Dataset Creation for Sales Call Transcripts
Creates anchor-positive-negative triplets for contrastive learning
"""

import pandas as pd
import numpy as np
import json
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContrastivePair:
    """Represents a contrastive learning pair"""
    anchor: str
    positive: str
    negative: str
    anchor_label: int
    positive_label: int
    negative_label: int
    metadata: Dict[str, Any]

class ContrastiveDatasetCreator:
    """Creates contrastive learning datasets from sales call transcripts"""
    
    def __init__(self, min_similarity_threshold: float = 0.3):
        self.min_similarity_threshold = min_similarity_threshold
        self.converted_transcripts = []
        self.non_converted_transcripts = []
        
    def load_transcripts(self, csv_path: str) -> None:
        """Load transcripts from CSV file"""
        logger.info(f"Loading transcripts from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Separate converted and non-converted transcripts
        self.converted_transcripts = df[df['conversion_label'] == 1]['call_transcript'].tolist()
        self.non_converted_transcripts = df[df['conversion_label'] == 0]['call_transcript'].tolist()
        
        logger.info(f"Loaded {len(self.converted_transcripts)} converted and {len(self.non_converted_transcripts)} non-converted transcripts")
        
    def create_positive_pairs(self, same_class_transcripts: List[str], n_pairs: int = None) -> List[Tuple[str, str]]:
        """Create positive pairs from same class transcripts"""
        if n_pairs is None:
            n_pairs = len(same_class_transcripts) // 2
            
        pairs = []
        for _ in range(n_pairs):
            if len(same_class_transcripts) < 2:
                break
                
            # Randomly select two different transcripts
            transcript1, transcript2 = random.sample(same_class_transcripts, 2)
            pairs.append((transcript1, transcript2))
            
        return pairs
    
    def create_negative_pairs(self, positive_transcripts: List[str], negative_transcripts: List[str], n_pairs: int = None) -> List[Tuple[str, str]]:
        """Create negative pairs from different class transcripts"""
        if n_pairs is None:
            n_pairs = min(len(positive_transcripts), len(negative_transcripts))
            
        pairs = []
        for _ in range(n_pairs):
            positive = random.choice(positive_transcripts)
            negative = random.choice(negative_transcripts)
            pairs.append((positive, negative))
            
        return pairs
    
    def create_triplets(self, n_triplets: int = None) -> List[ContrastivePair]:
        """Create anchor-positive-negative triplets"""
        if n_triplets is None:
            n_triplets = min(len(self.converted_transcripts), len(self.non_converted_transcripts)) * 2
            
        triplets = []
        
        # Create triplets with converted transcripts as anchors
        for _ in range(n_triplets // 2):
            if len(self.converted_transcripts) >= 2 and len(self.non_converted_transcripts) >= 1:
                anchor, positive = random.sample(self.converted_transcripts, 2)
                negative = random.choice(self.non_converted_transcripts)
                
                triplet = ContrastivePair(
                    anchor=anchor,
                    positive=positive,
                    negative=negative,
                    anchor_label=1,
                    positive_label=1,
                    negative_label=0,
                    metadata={
                        'anchor_type': 'converted',
                        'positive_type': 'converted',
                        'negative_type': 'non_converted'
                    }
                )
                triplets.append(triplet)
        
        # Create triplets with non-converted transcripts as anchors
        for _ in range(n_triplets // 2):
            if len(self.non_converted_transcripts) >= 2 and len(self.converted_transcripts) >= 1:
                anchor, positive = random.sample(self.non_converted_transcripts, 2)
                negative = random.choice(self.converted_transcripts)
                
                triplet = ContrastivePair(
                    anchor=anchor,
                    positive=positive,
                    negative=negative,
                    anchor_label=0,
                    positive_label=0,
                    negative_label=1,
                    metadata={
                        'anchor_type': 'non_converted',
                        'positive_type': 'non_converted',
                        'negative_type': 'converted'
                    }
                )
                triplets.append(triplet)
                
        logger.info(f"Created {len(triplets)} contrastive triplets")
        return triplets
    
    def create_hard_negative_mining_triplets(self, n_triplets: int = None) -> List[ContrastivePair]:
        """Create triplets with hard negative mining (more challenging negative examples)"""
        if n_triplets is None:
            n_triplets = min(len(self.converted_transcripts), len(self.non_converted_transcripts))
            
        triplets = []
        
        # For each converted transcript, find the most similar non-converted transcript
        for converted in self.converted_transcripts[:n_triplets//2]:
            # Simple similarity based on common words (can be enhanced with embeddings)
            best_negative = self._find_most_similar_negative(converted)
            
            if best_negative:
                positive = random.choice([t for t in self.converted_transcripts if t != converted])
                
                triplet = ContrastivePair(
                    anchor=converted,
                    positive=positive,
                    negative=best_negative,
                    anchor_label=1,
                    positive_label=1,
                    negative_label=0,
                    metadata={
                        'anchor_type': 'converted',
                        'positive_type': 'converted',
                        'negative_type': 'hard_negative'
                    }
                )
                triplets.append(triplet)
        
        # For each non-converted transcript, find the most similar converted transcript
        for non_converted in self.non_converted_transcripts[:n_triplets//2]:
            best_positive = self._find_most_similar_positive(non_converted)
            
            if best_positive:
                negative = random.choice([t for t in self.non_converted_transcripts if t != non_converted])
                
                triplet = ContrastivePair(
                    anchor=non_converted,
                    positive=negative,
                    negative=best_positive,
                    anchor_label=0,
                    positive_label=0,
                    negative_label=1,
                    metadata={
                        'anchor_type': 'non_converted',
                        'positive_type': 'non_converted',
                        'negative_type': 'hard_positive'
                    }
                )
                triplets.append(triplet)
                
        logger.info(f"Created {len(triplets)} hard negative mining triplets")
        return triplets
    
    def _find_most_similar_negative(self, anchor: str) -> str:
        """Find the most similar negative example to the anchor"""
        anchor_words = set(anchor.lower().split())
        best_similarity = 0
        best_negative = None
        
        for negative in self.non_converted_transcripts:
            negative_words = set(negative.lower().split())
            similarity = len(anchor_words.intersection(negative_words)) / len(anchor_words.union(negative_words))
            
            if similarity > best_similarity and similarity < 0.8:  # Not too similar
                best_similarity = similarity
                best_negative = negative
                
        return best_negative
    
    def _find_most_similar_positive(self, anchor: str) -> str:
        """Find the most similar positive example to the anchor"""
        anchor_words = set(anchor.lower().split())
        best_similarity = 0
        best_positive = None
        
        for positive in self.converted_transcripts:
            positive_words = set(positive.lower().split())
            similarity = len(anchor_words.intersection(positive_words)) / len(anchor_words.union(positive_words))
            
            if similarity > best_similarity and similarity < 0.8:  # Not too similar
                best_similarity = similarity
                best_positive = positive
                
        return best_positive
    
    def save_triplets(self, triplets: List[ContrastivePair], output_path: str) -> None:
        """Save triplets to JSON file"""
        data = []
        for triplet in triplets:
            data.append({
                'anchor': triplet.anchor,
                'positive': triplet.positive,
                'negative': triplet.negative,
                'anchor_label': triplet.anchor_label,
                'positive_label': triplet.positive_label,
                'negative_label': triplet.negative_label,
                'metadata': triplet.metadata
            })
            
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved {len(triplets)} triplets to {output_path}")
    
    def create_train_val_split(self, triplets: List[ContrastivePair], val_split: float = 0.2) -> Tuple[List[ContrastivePair], List[ContrastivePair]]:
        """Split triplets into training and validation sets"""
        train_triplets, val_triplets = train_test_split(
            triplets, 
            test_size=val_split, 
            random_state=42,
            stratify=[t.anchor_label for t in triplets]
        )
        
        logger.info(f"Split into {len(train_triplets)} training and {len(val_triplets)} validation triplets")
        return train_triplets, val_triplets

def main():
    """Example usage of the contrastive dataset creator"""
    creator = ContrastiveDatasetCreator()
    
    # Load transcripts
    creator.load_transcripts('data/sample_transcripts.csv')
    
    # Create regular triplets
    triplets = creator.create_triplets(n_triplets=100)
    
    # Create hard negative mining triplets
    hard_triplets = creator.create_hard_negative_mining_triplets(n_triplets=50)
    
    # Combine all triplets
    all_triplets = triplets + hard_triplets
    
    # Split into train/val
    train_triplets, val_triplets = creator.create_train_val_split(all_triplets)
    
    # Save datasets
    creator.save_triplets(train_triplets, 'data/train_triplets.json')
    creator.save_triplets(val_triplets, 'data/val_triplets.json')
    
    print(f"Created {len(all_triplets)} total triplets")
    print(f"Training: {len(train_triplets)}, Validation: {len(val_triplets)}")

if __name__ == "__main__":
    main() 