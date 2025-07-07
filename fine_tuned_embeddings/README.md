# 🎯 Sales Conversion AI System

An intelligent AI system that fine-tunes embedding models on sales call transcripts to improve conversion prediction accuracy using contrastive learning, domain-specific fine-tuning, and LangChain-powered pipelines.

## 🎯 Overview

This system leverages advanced NLP techniques to analyze sales call transcripts and predict customer conversion likelihood with high accuracy. It combines:

- **Contrastive Learning**: Creates anchor-positive-negative triplets for better embedding separation
- **Domain-Specific Fine-tuning**: Adapts pre-trained models to sales conversation patterns
- **LangChain Integration**: Modular pipeline orchestration for scalable deployment
- **Multiple Similarity Methods**: Cosine, Euclidean, MMR, and Hybrid scoring approaches

## 🎯 Target Users

- **B2B and B2C Sales Teams**: Improve conversion rates through data-driven insights
- **Sales Enablement Analysts**: Understand what drives successful conversions
- **Revenue Operations (RevOps) Teams**: Optimize sales processes and forecasting
- **AI Product Engineers**: Build intelligent CRM platforms and sales tools

## 🎯 Goals & Outcomes

| Goal | Metric | Target |
|------|--------|--------|
| Improve conversion prediction | AUC-ROC, F1 score | ≥ 0.80 |
| Capture sales nuances in embeddings | High separation in latent space | Clear clustering |
| Automate conversion scoring | Batch + real-time prediction | < 1 second per call |
| Compare fine-tuned vs generic embeddings | Metric delta | ≥ 15% improvement |

## 🛠️ Tech Stack

| Component | Tool/Library |
|-----------|--------------|
| Framework | Python + LangChain |
| Frontend/UI | Streamlit |
| Embeddings | sentence-transformers, OpenAI |
| Fine-Tuning | Contrastive loss (SimCLR, SupCon) |
| Model Training | PyTorch / HuggingFace Transformers |
| Evaluation | scikit-learn, custom ROC metrics |
| Storage | Local + vector DB (FAISS, Chroma) |

## 📁 Project Structure

```
sales_conversion_ai/
│
├── training/
│   ├── contrastive_dataset.py        # Create anchor-positive-negative pairs
│   ├── train_finetune.py             # Fine-tuning script
│   └── evaluation.py                 # Metric comparison
│
├── pipeline/
│   ├── embedder.py                   # Load and apply embeddings
│   ├── scorer.py                     # Similarity + scoring logic
│   └── langchain_chain.py            # LangChain integration
│
├── data/
│   ├── sample_transcripts.csv        # Sales transcripts with labels
│   ├── train_triplets.json           # Contrastive training pairs
│   └── val_triplets.json             # Validation pairs
│
├── models/
│   └── finetuned_sales_model/        # Saved fine-tuned model
│
├── ui/
│   └── app.py                        # Streamlit interface
│
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd sales_conversion_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

The system comes with sample sales call transcripts. To use your own data:

1. Prepare a CSV file with columns:
   - `call_transcript`: The sales call conversation text
   - `conversion_label`: 1 for converted, 0 for not converted
   - Additional metadata columns (optional)

2. Place your CSV file in the `data/` directory

### 3. Create Contrastive Dataset

```bash
cd training
python contrastive_dataset.py
```

This creates anchor-positive-negative triplets for contrastive learning.

### 4. Train Fine-tuned Model

```bash
python train_finetune.py
```

This fine-tunes the embedding model using contrastive learning.

### 5. Run Evaluation

```bash
python evaluation.py
```

This compares fine-tuned vs generic model performance.

### 6. Launch Streamlit UI

```bash
cd ui
streamlit run app.py
```

Access the web interface at `http://localhost:8501`

## 🔧 Core Features

### 1. 🔧 Domain-Specific Fine-Tuning Pipeline

**Input**: Sales call transcripts (with labels: converted / not converted)

**Strategy**:
- Sentence-pair contrastive learning (successful vs failed pairs)
- Fine-tune models like all-MiniLM, sales-BERT, or mpnet
- Output: Domain-specialized embedding model

**Tools**: HuggingFace Trainer or PyTorch Lightning + contrastive loss

### 2. ⚖️ Contrastive Learning Module

- Pairs of positive/negative sales examples
- Triplet or SupCon loss to maximize separation in vector space
- Hard negative mining for challenging examples

### 3. 🔄 LangChain Processing Pipeline

```
Sales Transcript → Embedder → Similarity Scorer → Conversion Classifier → Output
```

**Pipeline Components**:
- `CustomEmbedder`: Loads fine-tuned model
- `SimilarityScorer`: Outputs cosine similarity to converted prototype vectors
- `PredictionChain`: Thresholds similarity for probability scoring

### 4. 🧪 Evaluation & Comparison Framework

**Generic Embeddings**: Use base model (all-MiniLM)
**Fine-Tuned Embeddings**: Use new trained model

**Metrics**:
- Accuracy, F1 Score
- AUC-ROC
- Precision@K for ranked customer lists
- Latent space clustering visualization (UMAP, t-SNE)

### 5. 📈 Streamlit UI

- Upload new call transcript
- View predicted conversion likelihood
- Visualize embedding space & confidence level
- Compare generic vs. fine-tuned performance

## 📊 Usage Examples

### Basic Prediction

```python
from pipeline.embedder import SalesEmbedder
from pipeline.scorer import ConversionPredictor

# Load embedder
embedder = SalesEmbedder(model_path='models/finetuned_sales_model')

# Load training data and compute prototypes
transcripts = ["Agent: Hi, I'm calling about...", ...]
labels = [1, 0, 1, ...]
embedder.compute_prototypes(transcripts, labels)

# Predict conversion
test_transcript = "Agent: Hi, I'm calling about your recent inquiry..."
prediction = embedder.predict_conversion_score(test_transcript)
print(f"Conversion probability: {prediction['conversion_probability']:.2%}")
```

### Batch Processing

```python
# Process multiple transcripts
transcripts = ["Call 1...", "Call 2...", "Call 3..."]
predictions = embedder.predict_batch(transcripts)

for i, pred in enumerate(predictions):
    print(f"Call {i+1}: {pred['conversion_probability']:.2%} conversion probability")
```

### Model Comparison

```python
from training.evaluation import SalesConversionEvaluator

# Initialize evaluator
evaluator = SalesConversionEvaluator()
evaluator.load_models('models/finetuned_sales_model')

# Compare models
results = evaluator.compare_models(transcripts, labels)
print(f"Improvement in AUC-ROC: {results['improvement']['auc_roc']:.3f}")
```

## 📈 Performance Metrics

The system aims to achieve:

- **AUC-ROC ≥ 0.80**: High discriminative ability
- **F1 Score ≥ 0.80**: Balanced precision and recall
- **≥ 15% improvement** over generic embeddings
- **< 1 second** prediction time per call
- **≥ 90%** classification accuracy

## 🔍 Advanced Features

### Custom Scoring Methods

The system supports multiple similarity scoring approaches:

1. **Cosine Similarity**: Standard cosine similarity between embeddings
2. **Euclidean Distance**: Distance-based similarity
3. **MMR (Maximum Marginal Relevance)**: Diversity-aware similarity
4. **Hybrid**: Weighted combination of multiple methods

### Hard Negative Mining

Automatically identifies challenging negative examples for better contrastive learning:

```python
from training.contrastive_dataset import ContrastiveDatasetCreator

creator = ContrastiveDatasetCreator()
hard_triplets = creator.create_hard_negative_mining_triplets(n_triplets=50)
```

### Embedding Visualization

Visualize the embedding space to understand model behavior:

```python
from training.evaluation import SalesConversionEvaluator

evaluator = SalesConversionEvaluator()
evaluator.create_visualizations('evaluation_results')
```

## 🚀 Deployment

### Local Deployment

1. **Development**: Use Streamlit for interactive development
2. **Production**: Deploy as REST API using FastAPI or Flask
3. **Batch Processing**: Use scheduled jobs for large-scale analysis

### Cloud Deployment

- **AWS**: Deploy on EC2 with S3 for model storage
- **GCP**: Use Cloud Run with Cloud Storage
- **Azure**: Deploy on App Service with Blob Storage

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🔧 Configuration

### Environment Variables

```bash
# Model paths
FINETUNED_MODEL_PATH=models/finetuned_sales_model
GENERIC_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Training parameters
LEARNING_RATE=2e-5
BATCH_SIZE=16
EPOCHS=10

# Evaluation
EVAL_METRICS=accuracy,precision,recall,f1_score,auc_roc
```

### Model Configuration

```python
# Custom model configuration
config = {
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'max_length': 512,
    'embedding_dimension': 384,
    'margin': 1.0,  # Triplet loss margin
    'lambda_param': 0.5  # MMR parameter
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [LangChain](https://langchain.com/) for pipeline orchestration
- [Streamlit](https://streamlit.io/) for the web interface
- [Hugging Face](https://huggingface.co/) for transformer models

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Contact the development team
- Check the documentation in the `docs/` folder

## 🔮 Roadmap

- [ ] Multi-language support
- [ ] Real-time call analysis
- [ ] Integration with CRM systems
- [ ] Advanced conversation analytics
- [ ] A/B testing framework
- [ ] Mobile app development

---

**Built with ❤️ for better sales outcomes** 