# ⚖️ Indian Legal Document Retrieval System

A comprehensive legal document retrieval system for Indian law that compares 4 different similarity approaches side-by-side. Built for law researchers, legal professionals, and policy analysts.

## 🎯 Features

### 🔍 **4 Similarity Methods**
- **Cosine Similarity**: Classic semantic similarity using vector dot product
- **Euclidean Distance**: Geometric distance between embeddings
- **MMR (Maximal Marginal Relevance)**: Balances relevance and novelty
- **Hybrid**: Combines cosine similarity with legal entity matching

### 📄 **Document Processing**
- Support for PDF and Word documents
- Automatic text extraction and cleaning
- Intelligent section splitting for legal documents
- Embedding generation using sentence-transformers

### 📊 **Performance Evaluation**
- Precision@5, Recall, and F1-score metrics
- Diversity analysis (section and document level)
- Novelty and serendipity scoring
- Interactive performance dashboards

### 🖥️ **User Interface**
- Modern Streamlit web interface
- Side-by-side comparison of all methods
- Document upload and processing
- Export results to CSV
- Sample queries for testing

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **Document Processing**: PyMuPDF, python-docx
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Similarity Search**: FAISS
- **NLP**: spaCy for entity recognition
- **Visualization**: Plotly, Altair
- **Storage**: In-memory with FAISS indexing

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd legal_document_search

# Install dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Usage

1. **Initialize System**: Click "Initialize System" in the sidebar
2. **Upload Documents**: Upload PDF or Word legal documents
3. **Search**: Enter your legal query or select a sample query
4. **Compare Results**: View side-by-side results from all 4 methods
5. **Analyze Performance**: Check the metrics dashboard
6. **Export**: Download results as CSV

## 📁 Project Structure

```
legal_document_search/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── utils/
│   ├── document_processor.py      # Document processing and text extraction
│   ├── embedding_manager.py       # Embedding generation and similarity search
│   └── evaluation_metrics.py      # Performance evaluation metrics
├── data/
│   └── sample_legal_documents.py  # Sample documents for testing
└── documents/                     # Upload your legal documents here
```

## 🔧 Configuration

### Embedding Models
The system uses `all-MiniLM-L6-v2` by default. You can change this in `utils/embedding_manager.py`:

```python
embedding_manager = EmbeddingManager(model_name="your-preferred-model")
```

### Similarity Method Parameters
- **MMR Lambda**: Controls balance between relevance and diversity (default: 0.5)
- **Hybrid Weight**: Weight for cosine similarity vs entity matching (default: 0.6)

## 📊 Sample Queries

The system comes with 10 sample queries covering various legal domains:

1. "Income tax deduction for education expenses"
2. "GST rate for textile products and fabrics"
3. "Property registration process and required documents"
4. "Court fee structure for civil cases"
5. "Board composition requirements for public companies"
6. "Consumer rights for defective products"
7. "Minimum wages for construction workers"
8. "Environmental clearance for industrial projects"
9. "RBI regulations for banking operations"
10. "Arrest procedures and police powers"

## 📈 Performance Metrics

### Core Metrics
- **Precision@5**: Percentage of relevant documents in top 5 results
- **Recall@5**: Percentage of total relevant documents retrieved
- **F1-Score**: Harmonic mean of precision and recall

### Diversity Metrics
- **Section Diversity**: Unique sections in result set
- **Document Diversity**: Unique documents in result set
- **Novelty**: Introduction of new content areas
- **Serendipity**: Surprising but relevant results

## 🎯 Target Users

- **Law Researchers**: Compare different search approaches
- **Legal Professionals**: Quick access to relevant legal provisions
- **Policy Analysts**: Comprehensive legal document analysis
- **Government Teams**: Compliance and regulatory research
- **Law Students**: Learning and research tool

## 🔍 Supported Document Types

- **PDF**: Legal acts, regulations, court judgments
- **Word Documents**: Legal briefs, policy documents
- **Text Files**: Plain text legal documents

## 🚀 Advanced Features

### Custom Embedding Models
You can easily integrate other embedding models by modifying the `EmbeddingManager` class.

### Batch Processing
For large document collections, the system supports batch processing and indexing.

### Performance Optimization
- FAISS indexing for fast similarity search
- Efficient memory management
- Caching for repeated queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Sentence Transformers library for embeddings
- FAISS for efficient similarity search
- Streamlit for the web interface
- spaCy for natural language processing

## 📞 Support

For questions, issues, or feature requests, please open an issue on the repository.

---

**Built with ❤️ for the Indian legal community** 