import fitz  # PyMuPDF
from docx import Document
import re
import nltk
from typing import List, Dict, Tuple
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentProcessor:
    def __init__(self):
        self.supported_extensions = {'.pdf', '.docx', '.doc'}
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {e}")
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from Word document."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep legal references
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\–\—§]', '', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()
    
    def split_into_sections(self, text: str) -> List[Dict[str, str]]:
        """Split legal document into logical sections."""
        sections = []
        
        # Common legal section patterns
        section_patterns = [
            r'(?:Section|Article|Clause|Sub-section|Rule|Regulation)\s+(\d+[A-Za-z]*)',
            r'(?:CHAPTER|PART|DIVISION)\s+(\d+[A-Za-z]*)',
            r'(?:Schedule|Appendix|Annexure)\s+(\d+[A-Za-z]*)',
            r'(\d+\.\s+[A-Z][^.]*\.)',  # Numbered paragraphs
        ]
        
        # Split by patterns
        lines = text.split('\n')
        current_section = {"title": "Introduction", "content": "", "section_id": "intro"}
        section_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches any section pattern
            is_section_header = False
            for pattern in section_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous section if it has content
                    if current_section["content"].strip():
                        sections.append(current_section)
                    
                    # Start new section
                    section_id = match.group(1) if match.groups() else f"section_{section_counter}"
                    current_section = {
                        "title": line,
                        "content": "",
                        "section_id": section_id
                    }
                    section_counter += 1
                    is_section_header = True
                    break
            
            if not is_section_header:
                current_section["content"] += line + " "
        
        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        # If no sections found, split by paragraphs
        if len(sections) <= 1:
            sections = self.split_by_paragraphs(text)
        
        return sections
    
    def split_by_paragraphs(self, text: str) -> List[Dict[str, str]]:
        """Split text into paragraphs if no clear sections found."""
        paragraphs = text.split('\n\n')
        sections = []
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para and len(para) > 50:  # Only include substantial paragraphs
                sections.append({
                    "title": f"Paragraph {i+1}",
                    "content": para,
                    "section_id": f"para_{i+1}"
                })
        
        return sections
    
    def process_document(self, file_path: str) -> List[Dict[str, str]]:
        """Main method to process a document and return sections."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Extract text
        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            text = self.extract_text_from_docx(file_path)
        
        # Clean text
        text = self.clean_text(text)
        
        # Split into sections
        sections = self.split_into_sections(text)
        
        # Add document metadata
        for section in sections:
            section["document_name"] = os.path.basename(file_path)
            section["file_path"] = file_path
        
        return sections 