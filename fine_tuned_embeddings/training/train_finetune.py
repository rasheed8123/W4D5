"""
Fine-tuning Script for Sales Call Transcript Embeddings
Uses contrastive learning to improve conversion prediction accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.readers import InputExample
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContrastiveDataset(Dataset):
    """PyTorch Dataset for contrastive learning triplets"""
    
    def __init__(self, triplets: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        # Tokenize anchor, positive, and negative
        anchor_encoding = self.tokenizer(
            triplet['anchor'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        positive_encoding = self.tokenizer(
            triplet['positive'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        negative_encoding = self.tokenizer(
            triplet['negative'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(),
            'negative_input_ids': negative_encoding['input_ids'].squeeze(),
            'negative_attention_mask': negative_encoding['attention_mask'].squeeze(),
            'anchor_label': triplet['anchor_label'],
            'positive_label': triplet['positive_label'],
            'negative_label': triplet['negative_label']
        }

class TripletLoss(nn.Module):
    """Triplet loss for contrastive learning"""
    
    def __init__(self, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        anchor: anchor embeddings
        positive: positive sample embeddings
        negative: negative sample embeddings
        """
        # Compute distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Compute triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        return loss.mean()

class ContrastiveModel(nn.Module):
    """Model for contrastive learning with sales call transcripts"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        super(ContrastiveModel, self).__init__()
        self.encoder = SentenceTransformer(model_name)
        self.triplet_loss = TripletLoss(margin=1.0)
        
    def forward(self, anchor_input_ids, anchor_attention_mask,
                positive_input_ids, positive_attention_mask,
                negative_input_ids, negative_attention_mask):
        
        # Get embeddings
        anchor_emb = self.encoder({
            'input_ids': anchor_input_ids,
            'attention_mask': anchor_attention_mask
        })['sentence_embedding']
        
        positive_emb = self.encoder({
            'input_ids': positive_input_ids,
            'attention_mask': positive_attention_mask
        })['sentence_embedding']
        
        negative_emb = self.encoder({
            'input_ids': negative_input_ids,
            'attention_mask': negative_attention_mask
        })['sentence_embedding']
        
        # Compute triplet loss
        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        
        return loss, anchor_emb, positive_emb, negative_emb
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.encoder.encode(texts, batch_size=batch_size, show_progress_bar=True)

class SalesEmbeddingTrainer:
    """Trainer for fine-tuning embedding models on sales data"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 device: str = None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
    def load_data(self, train_path: str, val_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load training and validation data"""
        logger.info(f"Loading training data from {train_path}")
        with open(train_path, 'r') as f:
            train_triplets = json.load(f)
            
        logger.info(f"Loading validation data from {val_path}")
        with open(val_path, 'r') as f:
            val_triplets = json.load(f)
            
        logger.info(f"Loaded {len(train_triplets)} training and {len(val_triplets)} validation triplets")
        return train_triplets, val_triplets
    
    def train_with_sentence_transformers(self, train_triplets: List[Dict], 
                                       val_triplets: List[Dict],
                                       output_path: str,
                                       epochs: int = 10,
                                       batch_size: int = 16,
                                       learning_rate: float = 2e-5,
                                       warmup_steps: int = 1000):
        """Train using sentence-transformers library"""
        logger.info("Initializing sentence-transformers model")
        
        # Initialize model
        model = SentenceTransformer(self.model_name)
        
        # Prepare training examples
        train_examples = []
        for triplet in train_triplets:
            # Add anchor-positive pair
            train_examples.append(InputExample(
                texts=[triplet['anchor'], triplet['positive']], 
                label=1.0
            ))
            # Add anchor-negative pair
            train_examples.append(InputExample(
                texts=[triplet['anchor'], triplet['negative']], 
                label=0.0
            ))
        
        # Prepare validation examples
        val_examples = []
        for triplet in val_triplets:
            val_examples.append(InputExample(
                texts=[triplet['anchor'], triplet['positive']], 
                label=1.0
            ))
            val_examples.append(InputExample(
                texts=[triplet['anchor'], triplet['negative']], 
                label=0.0
            ))
        
        # Define loss function
        train_loss = losses.ContrastiveLoss(model)
        
        # Training arguments
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        evaluator = self._create_evaluator(val_examples)
        
        # Train the model
        logger.info("Starting training...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=100,
            warmup_steps=warmup_steps,
            output_path=output_path,
            optimizer_params={'lr': learning_rate},
            show_progress_bar=True
        )
        
        self.model = model
        logger.info(f"Training completed. Model saved to {output_path}")
        
    def train_with_pytorch(self, train_triplets: List[Dict], 
                          val_triplets: List[Dict],
                          output_path: str,
                          epochs: int = 10,
                          batch_size: int = 8,
                          learning_rate: float = 2e-5):
        """Train using PyTorch with custom triplet loss"""
        logger.info("Initializing PyTorch model")
        
        # Initialize model and tokenizer
        model = ContrastiveModel(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        model.to(self.device)
        
        # Create datasets
        train_dataset = ContrastiveDataset(train_triplets, tokenizer)
        val_dataset = ContrastiveDataset(val_triplets, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Training loop
        logger.info("Starting PyTorch training...")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                loss, _, _, _ = model(
                    batch['anchor_input_ids'], batch['anchor_attention_mask'],
                    batch['positive_input_ids'], batch['positive_attention_mask'],
                    batch['negative_input_ids'], batch['negative_attention_mask']
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    loss, _, _, _ = model(
                        batch['anchor_input_ids'], batch['anchor_attention_mask'],
                        batch['positive_input_ids'], batch['positive_attention_mask'],
                        batch['negative_input_ids'], batch['negative_attention_mask']
                    )
                    
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.encoder.save(output_path)
                logger.info(f"New best model saved to {output_path}")
        
        self.model = model.encoder
        logger.info("PyTorch training completed")
    
    def _create_evaluator(self, val_examples: List[InputExample]):
        """Create evaluator for sentence-transformers training"""
        from sentence_transformers.evaluation import BinaryClassificationEvaluator
        
        # Prepare evaluation data
        sentences1 = []
        sentences2 = []
        scores = []
        
        for example in val_examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        
        return BinaryClassificationEvaluator(sentences1, sentences2, scores)
    
    def evaluate_model(self, test_transcripts: List[str], test_labels: List[int]) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Encode test transcripts
        embeddings = self.model.encode(test_transcripts)
        
        # Compute similarities to converted and non-converted prototypes
        converted_embeddings = embeddings[test_labels == 1]
        non_converted_embeddings = embeddings[test_labels == 0]
        
        if len(converted_embeddings) == 0 or len(non_converted_embeddings) == 0:
            logger.warning("No examples in one of the classes")
            return {}
        
        # Compute prototype embeddings
        converted_prototype = np.mean(converted_embeddings, axis=0)
        non_converted_prototype = np.mean(non_converted_embeddings, axis=0)
        
        # Compute similarities
        converted_similarities = cosine_similarity(embeddings, [converted_prototype]).flatten()
        non_converted_similarities = cosine_similarity(embeddings, [non_converted_prototype]).flatten()
        
        # Predictions based on similarity
        predictions = (converted_similarities > non_converted_similarities).astype(int)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        # For AUC, use the difference in similarities as score
        scores = converted_similarities - non_converted_similarities
        auc = roc_auc_score(test_labels, scores)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc,
            'converted_similarity_mean': np.mean(converted_similarities[test_labels == 1]),
            'non_converted_similarity_mean': np.mean(non_converted_similarities[test_labels == 0])
        }

def main():
    """Main training script"""
    # Initialize trainer
    trainer = SalesEmbeddingTrainer()
    
    # Load data
    train_triplets, val_triplets = trainer.load_data(
        'data/train_triplets.json',
        'data/val_triplets.json'
    )
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Train with sentence-transformers (recommended)
    trainer.train_with_sentence_transformers(
        train_triplets=train_triplets,
        val_triplets=val_triplets,
        output_path='models/finetuned_sales_model',
        epochs=5,
        batch_size=16,
        learning_rate=2e-5
    )
    
    # Load test data for evaluation
    df = pd.read_csv('data/sample_transcripts.csv')
    test_transcripts = df['call_transcript'].tolist()
    test_labels = df['conversion_label'].values
    
    # Evaluate model
    metrics = trainer.evaluate_model(test_transcripts, test_labels)
    
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 