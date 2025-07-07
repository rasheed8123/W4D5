import os
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from .loader import DocumentLoader
from .classifier import DocumentClassifier, ClassificationResult
from .chunker import AdaptiveChunker, ChunkMetadata
from .embedder import DocumentEmbedder, EmbeddingResult
from .vector_store import VectorStore

@dataclass
class ProcessingResult:
    """Result of document processing pipeline."""
    document_id: str
    filename: str
    doc_type: str
    classification_confidence: float
    chunk_count: int
    processing_time: float
    chunks: List[Tuple[str, ChunkMetadata]]
    embedding_result: Optional[EmbeddingResult] = None
    vector_store_ids: Optional[List[str]] = None
    error: Optional[str] = None

class IntelligentChunkingPipeline:
    """Main orchestrator for the intelligent document chunking pipeline."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 vector_store_path: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_openai: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            model_path: Path to pre-trained classifier model
            vector_store_path: Path for ChromaDB storage
            embedding_model: Sentence transformer model name
            use_openai: Whether to use OpenAI embeddings
        """
        # Initialize components
        self.loader = DocumentLoader()
        self.classifier = DocumentClassifier(model_path=model_path)
        self.chunker = AdaptiveChunker()
        self.embedder = DocumentEmbedder(model_name=embedding_model, use_openai=use_openai)
        self.vector_store = VectorStore(persist_directory=vector_store_path)
        
        # Processing statistics
        self.processing_stats = {
            'total_documents': 0,
            'successful_documents': 0,
            'failed_documents': 0,
            'total_chunks': 0,
            'processing_times': [],
            'doc_type_distribution': {},
            'chunk_type_distribution': {}
        }
        
        # Results storage
        self.results: List[ProcessingResult] = []
    
    def process_document(self, file_path: str, auto_store: bool = True) -> ProcessingResult:
        """
        Process a single document through the entire pipeline.
        
        Args:
            file_path: Path to the document file
            auto_store: Whether to automatically store in vector database
            
        Returns:
            ProcessingResult with all processing information
        """
        start_time = time.time()
        document_id = f"doc_{int(time.time())}_{os.path.basename(file_path)}"
        
        try:
            # Step 1: Load document
            print(f"📄 Loading document: {file_path}")
            doc_data = self.loader.load_document(file_path)
            content = doc_data['content']
            metadata = doc_data['metadata']
            
            # Step 2: Classify document
            print(f"🏷️ Classifying document...")
            classification = self.classifier.classify_document(content, metadata.__dict__)
            
            # Step 3: Chunk document
            print(f"✂️ Chunking document using {classification.doc_type} strategy...")
            chunks = self.chunker.chunk_document(
                content=content,
                doc_type=classification.doc_type,
                metadata={'filename': metadata.filename, 'structure_tags': classification.structure_tags}
            )
            
            # Step 4: Generate embeddings
            embedding_result = None
            vector_store_ids = None
            
            if chunks and auto_store:
                print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
                chunk_texts = [chunk[0] for chunk in chunks]
                chunk_metadata = [chunk[1].__dict__ for chunk in chunks]
                
                embedding_result = self.embedder.embed_chunks(chunk_texts, chunk_metadata)
                
                # Step 5: Store in vector database
                print(f"💾 Storing chunks in vector database...")
                vector_store_ids = self.vector_store.add_chunks(
                    chunks=chunk_texts,
                    embeddings=embedding_result.embeddings,
                    metadata_list=chunk_metadata
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = ProcessingResult(
                document_id=document_id,
                filename=metadata.filename,
                doc_type=classification.doc_type,
                classification_confidence=classification.confidence,
                chunk_count=len(chunks),
                processing_time=processing_time,
                chunks=chunks,
                embedding_result=embedding_result,
                vector_store_ids=vector_store_ids
            )
            
            # Update statistics
            self._update_stats(result)
            self.results.append(result)
            
            print(f"✅ Document processed successfully in {processing_time:.2f}s")
            print(f"   - Type: {classification.doc_type} (confidence: {classification.confidence:.3f})")
            print(f"   - Chunks: {len(chunks)}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            print(f"❌ Error processing document: {error_msg}")
            
            # Create error result
            result = ProcessingResult(
                document_id=document_id,
                filename=os.path.basename(file_path),
                doc_type="unknown",
                classification_confidence=0.0,
                chunk_count=0,
                processing_time=processing_time,
                chunks=[],
                error=error_msg
            )
            
            # Update statistics
            self.processing_stats['failed_documents'] += 1
            self.results.append(result)
            
            return result
    
    def process_multiple_documents(self, file_paths: List[str], auto_store: bool = True) -> List[ProcessingResult]:
        """
        Process multiple documents.
        
        Args:
            file_paths: List of file paths to process
            auto_store: Whether to automatically store in vector database
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        for i, file_path in enumerate(file_paths):
            print(f"\n🔄 Processing document {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
            result = self.process_document(file_path, auto_store)
            results.append(result)
        
        return results
    
    def search_documents(self, query: str, n_results: int = 5, 
                        filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for documents using the vector store.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_single_chunk(query)
            
            # Search vector store
            results = self.vector_store.search(
                query=query,
                query_embedding=query_embedding,
                n_results=n_results,
                filter_metadata=filter_metadata
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def get_processing_stats(self) -> Dict:
        """Get comprehensive processing statistics."""
        stats = self.processing_stats.copy()
        
        # Calculate averages
        if stats['processing_times']:
            stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
            stats['min_processing_time'] = min(stats['processing_times'])
            stats['max_processing_time'] = max(stats['processing_times'])
        
        # Success rate
        if stats['total_documents'] > 0:
            stats['success_rate'] = stats['successful_documents'] / stats['total_documents']
        else:
            stats['success_rate'] = 0.0
        
        # Add vector store stats
        stats['vector_store_stats'] = self.vector_store.get_collection_stats()
        
        # Add embedding info
        stats['embedding_info'] = self.embedder.get_embedding_info()
        
        return stats
    
    def _update_stats(self, result: ProcessingResult):
        """Update processing statistics with a new result."""
        self.processing_stats['total_documents'] += 1
        
        if result.error:
            self.processing_stats['failed_documents'] += 1
        else:
            self.processing_stats['successful_documents'] += 1
            self.processing_stats['total_chunks'] += result.chunk_count
            self.processing_stats['processing_times'].append(result.processing_time)
            
            # Update doc type distribution
            doc_type = result.doc_type
            self.processing_stats['doc_type_distribution'][doc_type] = \
                self.processing_stats['doc_type_distribution'].get(doc_type, 0) + 1
            
            # Update chunk type distribution
            for chunk_content, chunk_metadata in result.chunks:
                chunk_type = chunk_metadata.chunk_type
                self.processing_stats['chunk_type_distribution'][chunk_type] = \
                    self.processing_stats['chunk_type_distribution'].get(chunk_type, 0) + 1
    
    def export_results(self, export_path: str):
        """Export processing results to JSON."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'pipeline_config': {
                    'embedding_model': self.embedder.model_name,
                    'use_openai': self.embedder.use_openai,
                    'vector_store_path': self.vector_store.persist_directory
                },
                'processing_stats': self.get_processing_stats(),
                'results': []
            }
            
            for result in self.results:
                result_data = {
                    'document_id': result.document_id,
                    'filename': result.filename,
                    'doc_type': result.doc_type,
                    'classification_confidence': result.classification_confidence,
                    'chunk_count': result.chunk_count,
                    'processing_time': result.processing_time,
                    'error': result.error,
                    'chunks': [
                        {
                            'content': chunk[0][:200] + "..." if len(chunk[0]) > 200 else chunk[0],
                            'metadata': chunk[1].__dict__
                        }
                        for chunk in result.chunks
                    ]
                }
                export_data['results'].append(result_data)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Exported results to: {export_path}")
            
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
            raise
    
    def reset_pipeline(self):
        """Reset the pipeline and clear all results."""
        self.results = []
        self.processing_stats = {
            'total_documents': 0,
            'successful_documents': 0,
            'failed_documents': 0,
            'total_chunks': 0,
            'processing_times': [],
            'doc_type_distribution': {},
            'chunk_type_distribution': {}
        }
        print("✅ Pipeline reset")
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration."""
        return {
            'loader': {
                'supported_formats': self.loader.get_supported_formats()
            },
            'classifier': self.classifier.get_classification_stats(),
            'embedder': self.embedder.get_embedding_info(),
            'vector_store': {
                'persist_directory': self.vector_store.persist_directory,
                'collection_name': self.vector_store.collection_name,
                'available_collections': self.vector_store.get_available_collections()
            }
        } 