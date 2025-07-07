import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os
import json
from datetime import datetime
import uuid

class VectorStore:
    """Handles vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "documents"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"✅ Loaded existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Document chunks for intelligent retrieval"}
                )
                print(f"✅ Created new collection: {self.collection_name}")
                
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")
            raise
    
    def add_chunks(self, chunks: List[str], embeddings: np.ndarray, 
                   metadata_list: List[Dict], ids: Optional[List[str]] = None) -> List[str]:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of text chunks
            embeddings: Embeddings for the chunks
            metadata_list: Metadata for each chunk
            ids: Optional custom IDs for chunks
            
        Returns:
            List of chunk IDs
        """
        if not chunks or len(chunks) == 0:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Prepare metadata
        processed_metadata = []
        for i, metadata in enumerate(metadata_list):
            processed_metadata.append({
                **metadata,
                'timestamp': datetime.now().isoformat(),
                'chunk_index': i
            })
        
        try:
            # Add to collection
            self.collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=processed_metadata,
                ids=ids
            )
            
            print(f"✅ Added {len(chunks)} chunks to vector store")
            return ids
            
        except Exception as e:
            print(f"❌ Error adding chunks to vector store: {e}")
            raise
    
    def search(self, query: str, query_embedding: np.ndarray, 
               n_results: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            query_embedding: Embedding of the query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with chunks and metadata
        """
        try:
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'chunk': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching vector store: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Get a specific chunk by ID."""
        try:
            results = self.collection.get(ids=[chunk_id])
            
            if results['documents'] and len(results['documents']) > 0:
                return {
                    'id': results['ids'][0],
                    'chunk': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting chunk by ID: {e}")
            return None
    
    def update_chunk(self, chunk_id: str, new_chunk: str, new_embedding: np.ndarray, 
                    new_metadata: Optional[Dict] = None):
        """Update an existing chunk."""
        try:
            # Get existing metadata
            existing = self.get_chunk_by_id(chunk_id)
            if not existing:
                raise Exception(f"Chunk with ID {chunk_id} not found")
            
            # Merge metadata
            metadata = existing['metadata'].copy()
            if new_metadata:
                metadata.update(new_metadata)
            metadata['updated_at'] = datetime.now().isoformat()
            
            # Update in collection
            self.collection.update(
                ids=[chunk_id],
                documents=[new_chunk],
                embeddings=[new_embedding.tolist()],
                metadatas=[metadata]
            )
            
            print(f"✅ Updated chunk: {chunk_id}")
            
        except Exception as e:
            print(f"❌ Error updating chunk: {e}")
            raise
    
    def delete_chunk(self, chunk_id: str):
        """Delete a chunk by ID."""
        try:
            self.collection.delete(ids=[chunk_id])
            print(f"✅ Deleted chunk: {chunk_id}")
        except Exception as e:
            print(f"❌ Error deleting chunk: {e}")
            raise
    
    def delete_chunks_by_filter(self, filter_metadata: Dict):
        """Delete chunks matching a filter."""
        try:
            # Get chunks matching filter
            results = self.collection.get(where=filter_metadata)
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"✅ Deleted {len(results['ids'])} chunks matching filter")
            else:
                print("No chunks found matching filter")
                
        except Exception as e:
            print(f"❌ Error deleting chunks by filter: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            # Get all documents
            results = self.collection.get()
            
            if not results['documents']:
                return {
                    'total_chunks': 0,
                    'documents': set(),
                    'chunk_types': {},
                    'date_range': None
                }
            
            # Calculate statistics
            total_chunks = len(results['documents'])
            documents = set()
            chunk_types = {}
            timestamps = []
            
            for metadata in results['metadatas']:
                # Document sources
                if 'source_document' in metadata:
                    documents.add(metadata['source_document'])
                
                # Chunk types
                chunk_type = metadata.get('chunk_type', 'unknown')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                # Timestamps
                if 'timestamp' in metadata:
                    timestamps.append(metadata['timestamp'])
            
            # Date range
            date_range = None
            if timestamps:
                timestamps.sort()
                date_range = {
                    'earliest': timestamps[0],
                    'latest': timestamps[-1]
                }
            
            return {
                'total_chunks': total_chunks,
                'documents': list(documents),
                'chunk_types': chunk_types,
                'date_range': date_range
            }
            
        except Exception as e:
            print(f"❌ Error getting collection stats: {e}")
            return {}
    
    def reset_collection(self):
        """Reset the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for intelligent retrieval"}
            )
            print(f"✅ Reset collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Error resetting collection: {e}")
            raise
    
    def export_collection(self, export_path: str):
        """Export collection data to JSON."""
        try:
            results = self.collection.get()
            
            export_data = {
                'collection_name': self.collection_name,
                'export_timestamp': datetime.now().isoformat(),
                'chunks': []
            }
            
            for i in range(len(results['documents'])):
                chunk_data = {
                    'id': results['ids'][i],
                    'chunk': results['documents'][i],
                    'metadata': results['metadatas'][i]
                }
                export_data['chunks'].append(chunk_data)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Exported collection to: {export_path}")
            
        except Exception as e:
            print(f"❌ Error exporting collection: {e}")
            raise
    
    def import_collection(self, import_path: str):
        """Import collection data from JSON."""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            chunks = []
            embeddings = []
            metadatas = []
            ids = []
            
            for chunk_data in import_data['chunks']:
                chunks.append(chunk_data['chunk'])
                metadatas.append(chunk_data['metadata'])
                ids.append(chunk_data['id'])
                # Note: embeddings would need to be regenerated
            
            # Add chunks (without embeddings for now)
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Imported {len(chunks)} chunks from: {import_path}")
            
        except Exception as e:
            print(f"❌ Error importing collection: {e}")
            raise
    
    def get_available_collections(self) -> List[str]:
        """Get list of available collections."""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"❌ Error getting collections: {e}")
            return []
    
    def switch_collection(self, collection_name: str):
        """Switch to a different collection."""
        try:
            self.collection = self.client.get_collection(name=collection_name)
            self.collection_name = collection_name
            print(f"✅ Switched to collection: {collection_name}")
        except Exception as e:
            print(f"❌ Error switching collection: {e}")
            raise 