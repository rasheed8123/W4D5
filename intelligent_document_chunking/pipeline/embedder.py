import numpy as np
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
import openai
import os
from dataclasses import dataclass

@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: np.ndarray
    model_name: str
    embedding_dim: int
    metadata: Dict

class DocumentEmbedder:
    """Handles embedding generation for document chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_openai: bool = False):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            use_openai: Whether to use OpenAI embeddings
        """
        self.model_name = model_name
        self.use_openai = use_openai
        self.model = None
        self.openai_client = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models."""
        try:
            # Initialize sentence transformer
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Loaded sentence transformer model: {self.model_name}")
        except Exception as e:
            print(f"❌ Error loading sentence transformer: {e}")
            # Fallback to default model
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                print("✅ Loaded fallback model: all-MiniLM-L6-v2")
            except Exception as e2:
                print(f"❌ Error loading fallback model: {e2}")
        
        # Initialize OpenAI if requested
        if self.use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized")
            else:
                print("⚠️ OpenAI API key not found. Using sentence transformers only.")
                self.use_openai = False
    
    def embed_chunks(self, chunks: List[str], metadata: Optional[List[Dict]] = None) -> EmbeddingResult:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of text chunks
            metadata: Optional metadata for each chunk
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not chunks:
            return EmbeddingResult(
                embeddings=np.array([]),
                model_name=self.model_name,
                embedding_dim=0,
                metadata={}
            )
        
        if self.use_openai and self.openai_client:
            return self._embed_with_openai(chunks, metadata)
        else:
            return self._embed_with_sentence_transformers(chunks, metadata)
    
    def _embed_with_sentence_transformers(self, chunks: List[str], metadata: Optional[List[Dict]] = None) -> EmbeddingResult:
        """Generate embeddings using sentence transformers."""
        if not self.model:
            raise Exception("Sentence transformer model not initialized")
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(chunks, show_progress_bar=True)
            
            # Convert to numpy array if needed
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.model_name,
                embedding_dim=embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                metadata={
                    'method': 'sentence_transformers',
                    'chunk_count': len(chunks),
                    'model_name': self.model_name
                }
            )
            
        except Exception as e:
            raise Exception(f"Error generating embeddings with sentence transformers: {e}")
    
    def _embed_with_openai(self, chunks: List[str], metadata: Optional[List[Dict]] = None) -> EmbeddingResult:
        """Generate embeddings using OpenAI."""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        try:
            embeddings = []
            
            # Process chunks in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            
            embeddings = np.array(embeddings)
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name="text-embedding-ada-002",
                embedding_dim=embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                metadata={
                    'method': 'openai',
                    'chunk_count': len(chunks),
                    'model_name': "text-embedding-ada-002"
                }
            )
            
        except Exception as e:
            raise Exception(f"Error generating embeddings with OpenAI: {e}")
    
    def embed_single_chunk(self, chunk: str) -> np.ndarray:
        """Generate embedding for a single chunk."""
        if self.use_openai and self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    input=[chunk],
                    model="text-embedding-ada-002"
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                print(f"OpenAI embedding failed, falling back to sentence transformers: {e}")
                self.use_openai = False
        
        if self.model:
            return self.model.encode([chunk])[0]
        else:
            raise Exception("No embedding model available")
    
    def get_embedding_info(self) -> Dict:
        """Get information about the current embedding setup."""
        info = {
            'model_name': self.model_name,
            'use_openai': self.use_openai,
            'sentence_transformer_loaded': self.model is not None,
            'openai_available': self.openai_client is not None
        }
        
        if self.model:
            info['embedding_dim'] = self.model.get_sentence_embedding_dimension()
        
        return info
    
    def change_model(self, model_name: str):
        """Change the sentence transformer model."""
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            print(f"✅ Changed to model: {model_name}")
        except Exception as e:
            print(f"❌ Error changing model: {e}")
    
    def toggle_openai(self, use_openai: bool):
        """Toggle OpenAI usage."""
        self.use_openai = use_openai
        if use_openai and not self.openai_client:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized")
            else:
                print("⚠️ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                self.use_openai = False 