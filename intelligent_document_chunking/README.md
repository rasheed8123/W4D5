# 🧠 Intelligent Document Chunking Agent

A comprehensive enterprise-grade document chunking system that automatically detects document types and applies tailored chunking strategies for optimal retrieval performance. Built for development teams, DevOps engineers, and AI/ML practitioners.

## 🎯 Features

### 🧠 **Intelligent Document Classification**
- **6 Document Types**: Technical docs, API references, troubleshooting tickets, policy documents, code tutorials, general docs
- **Hybrid Classification**: Combines rule-based patterns with ML models (TF-IDF + Random Forest)
- **Structure Detection**: Identifies headers, code blocks, tables, lists, and more
- **Confidence Scoring**: Provides classification confidence for quality assessment

### ✂️ **Adaptive Chunking Strategies**
- **Technical Docs**: Semantic + section header splitting
- **API References**: Method/function-wise chunking with context
- **Troubleshooting**: Q/A and step-based splitting
- **Policy Documents**: Paragraph + clause detection
- **Code Tutorials**: Code-aware + markdown block parsing
- **General Docs**: Recursive character splitting

### 🔗 **LangChain Integration**
- **Modular Pipeline**: DocumentLoader → Classifier → Chunker → Embedder → VectorStore
- **Custom Components**: Extensible architecture for custom chunking logic
- **Batch Processing**: Efficient handling of large document collections

### 📊 **Performance Monitoring**
- **Real-time Metrics**: Processing times, success rates, chunk distributions
- **Quality Metrics**: Classification accuracy, chunk size analysis
- **Retrieval Testing**: Query performance evaluation
- **Export Capabilities**: JSON export of processing results

### 🖥️ **Streamlit UI**
- **Document Upload**: Multi-format support (PDF, DOCX, MD, TXT, HTML)
- **Live Processing**: Real-time chunking with progress tracking
- **Interactive Dashboards**: Visual analytics and performance metrics
- **Search Interface**: Test retrieval performance with custom queries

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **Document Processing**: PyMuPDF, python-docx, markdown
- **Classification**: scikit-learn, transformers
- **Embeddings**: sentence-transformers, OpenAI (optional)
- **Vector Store**: ChromaDB
- **Chunking**: LangChain text splitters + custom strategies
- **Visualization**: Plotly, Altair

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd intelligent_document_chunking

# Install dependencies
pip install -r requirements.txt

# Optional: Install spaCy model for enhanced NER
python -m spacy download en_core_web_sm
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Usage

1. **Initialize Pipeline**: Click "Initialize Pipeline" in sidebar
2. **Upload Documents**: Drag & drop files or use file picker
3. **Process Documents**: Click "Process Documents" to start chunking
4. **View Results**: Check processing results and chunk previews
5. **Monitor Performance**: Use metrics dashboard for insights
6. **Test Retrieval**: Use search interface to test chunk quality

## 📁 Project Structure

```
intelligent_document_chunking/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── pipeline/
│   ├── __init__.py
│   ├── loader.py                   # Document loading & preprocessing
│   ├── classifier.py               # Document type classification
│   ├── chunker.py                  # Adaptive chunking strategies
│   ├── embedder.py                 # Embedding generation
│   ├── vector_store.py             # ChromaDB operations
│   └── orchestrator.py             # Main pipeline coordinator
├── data/
│   ├── raw/                        # Uploaded documents
│   └── processed/                  # Chunked outputs
├── models/
│   └── classifier_model.pkl        # Trained classifier (optional)
└── chroma_db/                      # Vector database storage
```

## 🔧 Configuration

### Document Types & Strategies

| Document Type | Chunking Strategy | Use Case |
|---------------|-------------------|----------|
| Technical Docs | Semantic + Headers | API docs, user guides |
| API References | Method-wise | API documentation |
| Troubleshooting | Q/A + Steps | Support tickets, FAQs |
| Policy Documents | Paragraph + Clauses | Legal docs, policies |
| Code Tutorials | Code-aware | Programming guides |
| General Docs | Recursive | Miscellaneous content |

### Embedding Models

```python
# Available sentence transformer models
"all-MiniLM-L6-v2"      # Fast, good quality (default)
"all-mpnet-base-v2"     # Higher quality, slower
"multi-qa-MiniLM-L6-cos-v1"  # Optimized for Q&A
```

### Vector Store Options

- **ChromaDB**: Local persistence, fast retrieval
- **Collection Management**: Multiple collections for different domains
- **Metadata Filtering**: Filter by document type, chunk type, etc.

## 📊 Performance Metrics

### Classification Metrics
- **Accuracy**: ≥ 90% classification accuracy
- **Confidence Scoring**: Rule-based + ML ensemble
- **Structure Detection**: Headers, code blocks, tables, lists

### Chunking Quality
- **Chunk Size Distribution**: Optimal 100-500 words
- **Semantic Coherence**: Context preservation
- **Retrieval Precision**: ≥ 15% improvement over baseline

### Processing Performance
- **Batch Processing**: Efficient multi-document handling
- **Memory Management**: Optimized for large collections
- **Caching**: Embedding and classification caching

## 🎯 Target Users

### **Internal Development Teams**
- Process technical documentation
- Build knowledge bases
- Improve code documentation

### **DevOps & Knowledge Management**
- Automate documentation processing
- Maintain up-to-date knowledge bases
- Streamline information retrieval

### **Technical Writers**
- Optimize content structure
- Improve searchability
- Maintain consistency

### **AI/ML Teams**
- Build RAG systems
- Train custom models
- Evaluate retrieval performance

## 🔍 Supported Document Types

- **PDF**: Technical manuals, research papers, reports
- **Word Documents**: Policy documents, user guides
- **Markdown**: Documentation, README files, tutorials
- **Text Files**: Logs, plain text documents
- **HTML**: Web content, documentation sites

## 🚀 Advanced Features

### Custom Chunking Strategies
```python
# Add custom chunking strategy
def custom_chunking_strategy(content, metadata):
    # Your custom logic here
    return chunks

# Register with pipeline
pipeline.chunker.chunking_strategies['custom_type'] = custom_chunking_strategy
```

### Batch Processing
```python
# Process multiple documents
results = pipeline.process_multiple_documents(file_paths)

# Export results
pipeline.export_results("results.json")
```

### Performance Optimization
- **Parallel Processing**: Multi-threaded document processing
- **Memory Efficient**: Streaming for large documents
- **Caching**: Embedding and classification caching

## 📈 Monitoring & Analytics

### Real-time Dashboards
- **Processing Metrics**: Success rates, processing times
- **Quality Metrics**: Classification confidence, chunk distributions
- **Performance Trends**: Historical analysis

### Export Capabilities
- **JSON Export**: Complete processing results
- **CSV Reports**: Metrics and statistics
- **Chunk Analysis**: Detailed chunk metadata

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain for the text splitting framework
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Streamlit for the web interface
- scikit-learn for classification

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the documentation
- Review example usage

---

**Built with ❤️ for the enterprise AI community** 