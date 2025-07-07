import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)
from langchain.schema import Document
import ast
import tokenize
from io import StringIO

@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    chunk_type: str
    source_document: str
    start_position: int
    end_position: int
    word_count: int
    section_header: Optional[str] = None
    code_language: Optional[str] = None
    confidence_score: Optional[float] = None

class AdaptiveChunker:
    """Adaptive document chunking based on document type and structure."""
    
    def __init__(self):
        self.chunking_strategies = {
            'technical_docs': self._chunk_technical_docs,
            'api_references': self._chunk_api_references,
            'troubleshooting_tickets': self._chunk_troubleshooting_tickets,
            'policy_documents': self._chunk_policy_documents,
            'code_tutorials': self._chunk_code_tutorials,
            'general_docs': self._chunk_general_docs
        }
        
        # Initialize LangChain splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        )
    
    def chunk_document(self, content: str, doc_type: str, metadata: Optional[Dict] = None) -> List[Tuple[str, ChunkMetadata]]:
        """
        Chunk a document using the appropriate strategy.
        
        Args:
            content: Document content
            doc_type: Classified document type
            metadata: Document metadata
            
        Returns:
            List of (chunk_content, chunk_metadata) tuples
        """
        if doc_type not in self.chunking_strategies:
            doc_type = 'general_docs'
        
        strategy_func = self.chunking_strategies[doc_type]
        return strategy_func(content, metadata)
    
    def _chunk_technical_docs(self, content: str, metadata: Optional[Dict] = None) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk technical documentation using semantic + section headers."""
        chunks = []
        
        # First, split by markdown headers
        if 'has_headers' in metadata.get('structure_tags', []):
            try:
                md_chunks = self.markdown_splitter.split_text(content)
                for i, chunk in enumerate(md_chunks):
                    chunk_metadata = ChunkMetadata(
                        chunk_id=f"tech_{i}",
                        chunk_type="semantic_section",
                        source_document=metadata.get('filename', 'unknown'),
                        start_position=content.find(chunk.page_content),
                        end_position=content.find(chunk.page_content) + len(chunk.page_content),
                        word_count=len(chunk.page_content.split()),
                        section_header=chunk.metadata.get('Header 1', chunk.metadata.get('Header 2', None))
                    )
                    chunks.append((chunk.page_content, chunk_metadata))
            except:
                # Fallback to recursive splitting
                pass
        
        # If no headers or markdown splitting failed, use recursive splitting
        if not chunks:
            split_chunks = self.recursive_splitter.split_text(content)
            for i, chunk in enumerate(split_chunks):
                chunk_metadata = ChunkMetadata(
                    chunk_id=f"tech_{i}",
                    chunk_type="semantic",
                    source_document=metadata.get('filename', 'unknown'),
                    start_position=content.find(chunk),
                    end_position=content.find(chunk) + len(chunk),
                    word_count=len(chunk.split())
                )
                chunks.append((chunk, chunk_metadata))
        
        return chunks
    
    def _chunk_api_references(self, content: str, metadata: Optional[Dict] = None) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk API references by method/function."""
        chunks = []
        
        # Split by API endpoints and methods
        api_patterns = [
            r'(GET|POST|PUT|DELETE|PATCH)\s+([^\n]+)',
            r'(def|function)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            r'(class)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'(endpoint|api|route)\s*[:=]\s*([^\n]+)'
        ]
        
        # Find all API-related sections
        api_sections = []
        for pattern in api_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 200)  # Include some context
                end = min(len(content), match.end() + 500)  # Include method details
                api_sections.append((start, end, match.group()))
        
        # Sort by position and merge overlapping sections
        api_sections.sort(key=lambda x: x[0])
        merged_sections = []
        for start, end, method in api_sections:
            if not merged_sections or start > merged_sections[-1][1]:
                merged_sections.append((start, end, method))
            else:
                # Merge overlapping sections
                merged_sections[-1] = (merged_sections[-1][0], max(merged_sections[-1][1], end), merged_sections[-1][2])
        
        # Create chunks from API sections
        for i, (start, end, method) in enumerate(merged_sections):
            chunk_content = content[start:end].strip()
            if len(chunk_content) > 50:  # Only include substantial chunks
                chunk_metadata = ChunkMetadata(
                    chunk_id=f"api_{i}",
                    chunk_type="api_method",
                    source_document=metadata.get('filename', 'unknown'),
                    start_position=start,
                    end_position=end,
                    word_count=len(chunk_content.split()),
                    section_header=method
                )
                chunks.append((chunk_content, chunk_metadata))
        
        # If no API sections found, fall back to semantic splitting
        if not chunks:
            return self._chunk_general_docs(content, metadata)
        
        return chunks
    
    def _chunk_troubleshooting_tickets(self, content: str, metadata: Optional[Dict] = None) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk troubleshooting tickets by Q/A or step-based splitting."""
        chunks = []
        
        # Split by common troubleshooting patterns
        patterns = [
            r'(Problem|Issue|Error|Bug):\s*([^\n]+(?:\n(?!Solution|Fix|Resolution)[^\n]+)*)',
            r'(Solution|Fix|Resolution):\s*([^\n]+(?:\n(?!Problem|Issue|Error|Bug)[^\n]+)*)',
            r'(Step \d+[\.:]?\s*[^\n]+(?:\n(?!Step \d+)[^\n]+)*)',
            r'(Cause|Root Cause):\s*([^\n]+(?:\n(?!Solution|Fix)[^\n]+)*)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for i, match in enumerate(matches):
                chunk_content = match.group().strip()
                if len(chunk_content) > 30:
                    chunk_metadata = ChunkMetadata(
                        chunk_id=f"trouble_{i}",
                        chunk_type="qa_step",
                        source_document=metadata.get('filename', 'unknown'),
                        start_position=match.start(),
                        end_position=match.end(),
                        word_count=len(chunk_content.split()),
                        section_header=match.group(1) if match.groups() else None
                    )
                    chunks.append((chunk_content, chunk_metadata))
        
        # If no structured patterns found, split by paragraphs
        if not chunks:
            paragraphs = re.split(r'\n\s*\n', content)
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) > 50:
                    chunk_metadata = ChunkMetadata(
                        chunk_id=f"trouble_para_{i}",
                        chunk_type="paragraph",
                        source_document=metadata.get('filename', 'unknown'),
                        start_position=content.find(para),
                        end_position=content.find(para) + len(para),
                        word_count=len(para.split())
                    )
                    chunks.append((para, chunk_metadata))
        
        return chunks
    
    def _chunk_policy_documents(self, content: str, metadata: Optional[Dict] = None) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk policy documents by paragraph + clause detection."""
        chunks = []
        
        # Split by policy-specific patterns
        policy_patterns = [
            r'(Section \d+[\.:]?\s*[^\n]+(?:\n(?!Section \d+)[^\n]+)*)',
            r'(Clause \d+[\.:]?\s*[^\n]+(?:\n(?!Clause \d+)[^\n]+)*)',
            r'(Policy \d+[\.:]?\s*[^\n]+(?:\n(?!Policy \d+)[^\n]+)*)',
            r'(Article \d+[\.:]?\s*[^\n]+(?:\n(?!Article \d+)[^\n]+)*)'
        ]
        
        for pattern in policy_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for i, match in enumerate(matches):
                chunk_content = match.group().strip()
                if len(chunk_content) > 50:
                    chunk_metadata = ChunkMetadata(
                        chunk_id=f"policy_{i}",
                        chunk_type="policy_clause",
                        source_document=metadata.get('filename', 'unknown'),
                        start_position=match.start(),
                        end_position=match.end(),
                        word_count=len(chunk_content.split()),
                        section_header=match.group(1) if match.groups() else None
                    )
                    chunks.append((chunk_content, chunk_metadata))
        
        # If no policy patterns found, split by paragraphs
        if not chunks:
            paragraphs = re.split(r'\n\s*\n', content)
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) > 100:  # Policy paragraphs are usually longer
                    chunk_metadata = ChunkMetadata(
                        chunk_id=f"policy_para_{i}",
                        chunk_type="policy_paragraph",
                        source_document=metadata.get('filename', 'unknown'),
                        start_position=content.find(para),
                        end_position=content.find(para) + len(para),
                        word_count=len(para.split())
                    )
                    chunks.append((para, chunk_metadata))
        
        return chunks
    
    def _chunk_code_tutorials(self, content: str, metadata: Optional[Dict] = None) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk code tutorials using code-aware + markdown block parser."""
        chunks = []
        
        # Split by code blocks first
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        code_matches = list(re.finditer(code_block_pattern, content, re.DOTALL))
        
        # Process code blocks and surrounding text
        last_end = 0
        for i, match in enumerate(code_matches):
            # Add text before code block
            if match.start() > last_end:
                text_before = content[last_end:match.start()].strip()
                if len(text_before) > 30:
                    chunk_metadata = ChunkMetadata(
                        chunk_id=f"tutorial_text_{i}",
                        chunk_type="tutorial_text",
                        source_document=metadata.get('filename', 'unknown'),
                        start_position=last_end,
                        end_position=match.start(),
                        word_count=len(text_before.split())
                    )
                    chunks.append((text_before, chunk_metadata))
            
            # Add code block
            code_content = match.group(0)
            code_language = match.group(1) if match.group(1) else 'text'
            
            chunk_metadata = ChunkMetadata(
                chunk_id=f"tutorial_code_{i}",
                chunk_type="code_block",
                source_document=metadata.get('filename', 'unknown'),
                start_position=match.start(),
                end_position=match.end(),
                word_count=len(code_content.split()),
                code_language=code_language
            )
            chunks.append((code_content, chunk_metadata))
            
            last_end = match.end()
        
        # Add remaining text after last code block
        if last_end < len(content):
            text_after = content[last_end:].strip()
            if len(text_after) > 30:
                chunk_metadata = ChunkMetadata(
                    chunk_id=f"tutorial_text_final",
                    chunk_type="tutorial_text",
                    source_document=metadata.get('filename', 'unknown'),
                    start_position=last_end,
                    end_position=len(content),
                    word_count=len(text_after.split())
                )
                chunks.append((text_after, chunk_metadata))
        
        # If no code blocks found, use markdown splitting
        if not chunks:
            try:
                md_chunks = self.markdown_splitter.split_text(content)
                for i, chunk in enumerate(md_chunks):
                    chunk_metadata = ChunkMetadata(
                        chunk_id=f"tutorial_md_{i}",
                        chunk_type="markdown_section",
                        source_document=metadata.get('filename', 'unknown'),
                        start_position=content.find(chunk.page_content),
                        end_position=content.find(chunk.page_content) + len(chunk.page_content),
                        word_count=len(chunk.page_content.split()),
                        section_header=chunk.metadata.get('Header 1', chunk.metadata.get('Header 2', None))
                    )
                    chunks.append((chunk.page_content, chunk_metadata))
            except:
                # Fallback to general chunking
                return self._chunk_general_docs(content, metadata)
        
        return chunks
    
    def _chunk_general_docs(self, content: str, metadata: Optional[Dict] = None) -> List[Tuple[str, ChunkMetadata]]:
        """General chunking strategy using recursive character splitting."""
        chunks = []
        
        split_chunks = self.recursive_splitter.split_text(content)
        for i, chunk in enumerate(split_chunks):
            chunk_metadata = ChunkMetadata(
                chunk_id=f"general_{i}",
                chunk_type="semantic",
                source_document=metadata.get('filename', 'unknown'),
                start_position=content.find(chunk),
                end_position=content.find(chunk) + len(chunk),
                word_count=len(chunk.split())
            )
            chunks.append((chunk, chunk_metadata))
        
        return chunks
    
    def get_chunking_stats(self, chunks: List[Tuple[str, ChunkMetadata]]) -> Dict:
        """Get statistics about the chunking process."""
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'total_words': sum(chunk[1].word_count for chunk in chunks),
            'avg_chunk_size': sum(chunk[1].word_count for chunk in chunks) / len(chunks),
            'chunk_types': {},
            'size_distribution': {
                'small': 0,    # < 100 words
                'medium': 0,   # 100-500 words
                'large': 0     # > 500 words
            }
        }
        
        for chunk_content, chunk_metadata in chunks:
            # Count chunk types
            chunk_type = chunk_metadata.chunk_type
            stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
            
            # Count size distribution
            word_count = chunk_metadata.word_count
            if word_count < 100:
                stats['size_distribution']['small'] += 1
            elif word_count < 500:
                stats['size_distribution']['medium'] += 1
            else:
                stats['size_distribution']['large'] += 1
        
        return stats 