import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import spacy
from typing import List, Dict, Tuple, Optional
import os
import pickle

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager with specified model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = []
        self.documents = []
        self.index = None
        self.nlp = None
        
        # Try to load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Warning: spaCy model could not be loaded. NER features will be disabled.")
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the embedding manager.
        
        Args:
            documents: List of document dictionaries with 'content' key
        """
        texts = [doc['content'] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Convert to float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.embeddings.append(embeddings)
        self.documents.extend(documents)
        
        # Update FAISS index
        self._update_index()
    
    def _update_index(self):
        """Update FAISS index with all embeddings."""
        if not self.embeddings:
            return
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(self.embeddings)
        
        # Create FAISS index
        dimension = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(all_embeddings)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        embedding = self.model.encode([text])
        embedding = embedding.astype('float32')
        faiss.normalize_L2(embedding)
        return embedding
    
    def extract_legal_entities(self, text: str) -> List[str]:
        """Extract legal entities from text using spaCy NER."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        # Legal entity patterns
        legal_patterns = [
            'LAW', 'ORG', 'PERSON', 'GPE', 'DATE', 'MONEY', 'PERCENT'
        ]
        
        for ent in doc.ents:
            if ent.label_ in legal_patterns:
                entities.append(ent.text.lower())
        
        return entities
    
    def cosine_similarity_search(self, query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Perform cosine similarity search."""
        if not self.index or not self.documents:
            return []
        
        query_embedding = self.get_embedding(query)
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append((idx, float(score), self.documents[idx]))
        
        return results
    
    def euclidean_distance_search(self, query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Perform Euclidean distance search."""
        if not self.index or not self.documents:
            return []
        
        query_embedding = self.get_embedding(query)
        
        # Calculate Euclidean distances
        all_embeddings = np.vstack(self.embeddings)
        distances = np.linalg.norm(all_embeddings - query_embedding, axis=1)
        
        # Get top k results (lower distance is better)
        indices = np.argsort(distances)[:k]
        
        results = []
        for idx in indices:
            if idx < len(self.documents):
                # Convert distance to similarity score (1 / (1 + distance))
                similarity = 1 / (1 + distances[idx])
                results.append((idx, float(similarity), self.documents[idx]))
        
        return results
    
    def mmr_search(self, query: str, k: int = 5, lambda_param: float = 0.5) -> List[Tuple[int, float, Dict]]:
        """
        Perform Maximal Marginal Relevance search.
        
        Args:
            query: Search query
            k: Number of results to return
            lambda_param: Balance between relevance (lambda) and diversity (1-lambda)
        """
        if not self.index or not self.documents:
            return []
        
        query_embedding = self.get_embedding(query)
        all_embeddings = np.vstack(self.embeddings)
        
        # Get initial relevance scores
        relevance_scores = np.dot(all_embeddings, query_embedding.T).flatten()
        
        # Initialize results
        selected_indices = []
        selected_embeddings = []
        
        # Select first document (highest relevance)
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        selected_embeddings.append(all_embeddings[first_idx])
        
        # Select remaining documents using MMR
        for _ in range(k - 1):
            mmr_scores = []
            
            for i in range(len(all_embeddings)):
                if i in selected_indices:
                    mmr_scores.append(-1)  # Already selected
                    continue
                
                # Relevance score
                relevance = relevance_scores[i]
                
                # Diversity score (max similarity to already selected)
                if selected_embeddings:
                    similarities = np.dot(all_embeddings[i], np.array(selected_embeddings).T)
                    diversity = 1 - np.max(similarities)
                else:
                    diversity = 1
                
                # MMR score
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                mmr_scores.append(mmr_score)
            
            # Select document with highest MMR score
            next_idx = np.argmax(mmr_scores)
            selected_indices.append(next_idx)
            selected_embeddings.append(all_embeddings[next_idx])
        
        # Return results
        results = []
        for idx in selected_indices:
            if idx < len(self.documents):
                similarity = float(relevance_scores[idx])
                results.append((idx, similarity, self.documents[idx]))
        
        return results
    
    def hybrid_search(self, query: str, k: int = 5, cosine_weight: float = 0.6) -> List[Tuple[int, float, Dict]]:
        """
        Perform hybrid search combining cosine similarity and legal entity matching.
        
        Args:
            query: Search query
            k: Number of results to return
            cosine_weight: Weight for cosine similarity (1 - cosine_weight for entity matching)
        """
        if not self.index or not self.documents:
            return []
        
        # Get cosine similarity results
        cosine_results = self.cosine_similarity_search(query, k * 2)  # Get more candidates
        
        # Extract entities from query
        query_entities = set(self.extract_legal_entities(query))
        
        # Calculate hybrid scores
        hybrid_scores = []
        for idx, cosine_score, doc in cosine_results:
            # Extract entities from document
            doc_entities = set(self.extract_legal_entities(doc['content']))
            
            # Calculate entity overlap
            if query_entities and doc_entities:
                entity_overlap = len(query_entities.intersection(doc_entities)) / len(query_entities.union(doc_entities))
            else:
                entity_overlap = 0
            
            # Calculate hybrid score
            hybrid_score = cosine_weight * cosine_score + (1 - cosine_weight) * entity_overlap
            hybrid_scores.append((idx, hybrid_score, doc))
        
        # Sort by hybrid score and return top k
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:k]
    
    def save_index(self, filepath: str):
        """Save the FAISS index and documents to disk."""
        if self.index:
            faiss.write_index(self.index, filepath + ".index")
        
        with open(filepath + ".docs", "wb") as f:
            pickle.dump(self.documents, f)
    
    def load_index(self, filepath: str):
        """Load the FAISS index and documents from disk."""
        if os.path.exists(filepath + ".index"):
            self.index = faiss.read_index(filepath + ".index")
        
        if os.path.exists(filepath + ".docs"):
            with open(filepath + ".docs", "rb") as f:
                self.documents = pickle.load(f)
            
            # Reconstruct embeddings for other methods
            if self.documents:
                texts = [doc['content'] for doc in self.documents]
                embeddings = self.model.encode(texts, show_progress_bar=False)
                embeddings = embeddings.astype('float32')
                faiss.normalize_L2(embeddings)
                self.embeddings = [embeddings] 