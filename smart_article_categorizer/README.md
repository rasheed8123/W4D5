# Smart Article Categorizer

An automated system to classify news articles into 6 categories using 4 different embedding approaches, with a Streamlit web interface for real-time predictions and model comparison.

## Features
- **Data Processing:** Cleans and preprocesses news articles from CSV files.
- **Embeddings:** Supports GloVe, BERT, Sentence-BERT, and OpenAI embeddings.
- **Classification:** Trains Logistic Regression classifiers for each embedding type.
- **Evaluation:** Compares models using accuracy, precision, recall, and F1-score.
- **Web Interface:** Real-time article classification and side-by-side model comparison with visualizations.

## Categories
- Tech
- Finance
- Healthcare
- Sports
- Politics
- Entertainment

## Project Structure
```
smart_article_categorizer/
├── data/
│   └── articles.csv
├── models/
│   └── logistic_regression/
├── embeddings/
│   └── glove.6B.300d.txt
├── utils/
│   ├── preprocess.py
│   ├── embedding_models.py
│   └── model_utils.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download GloVe embeddings:**
   - Place `glove.6B.300d.txt` in `embeddings/` (download from [GloVe website](https://nlp.stanford.edu/projects/glove/)).
3. **Prepare data:**
   - Place your `articles.csv` in `data/` with columns: `text`, `category`.
4. **Train models:**
   - Use provided scripts (to be implemented) to train and save classifiers for each embedding type.
5. **Set OpenAI API Key:**
   - Set the `OPENAI_API_KEY` environment variable for OpenAI embeddings.

## Usage
1. **Run the Streamlit app:**
   ```bash
   streamlit run smart_article_categorizer/streamlit_app.py
   ```
2. **Web Interface:**
   - Paste an article in the text area.
   - View predictions and confidence scores from all 4 models.
   - Compare model performance visually.

## Performance Metrics
- Accuracy, Precision, Recall, F1-score for each model.
- Model comparison table and bar chart.

## Optional
- 2D embedding visualization using PCA/t-SNE.

## License
MIT 