import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.preprocess import clean_text, CATEGORY_LABELS
from utils.embedding_models import GloveEmbedder, BertEmbedder, SBertEmbedder, OpenAIEmbedder
from utils.model_utils import load_model
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Smart Article Categorizer", layout="wide")
st.title("📰 Smart Article Categorizer")

# Load models and embedders with error handling
@st.cache_resource
def load_all_models():
    embedders = {}
    models = {}
    
    # Try to load GloVe
    try:
        embedders['GloVe'] = GloveEmbedder("embeddings/glove.6B.300d.txt")
        models['GloVe'] = load_model("models/logistic_regression/glove.joblib")
    except Exception as e:
        st.warning(f"GloVe model not available: {e}")
    
    # Try to load BERT
    try:
        embedders['BERT'] = BertEmbedder()
        models['BERT'] = load_model("models/logistic_regression/bert.joblib")
    except Exception as e:
        st.warning(f"BERT model not available: {e}")
    
    # Try to load SBERT
    try:
        embedders['SBERT'] = SBertEmbedder()
        models['SBERT'] = load_model("models/logistic_regression/sbert.joblib")
    except Exception as e:
        st.warning(f"SBERT model not available: {e}")
    
    # Try to load OpenAI
    try:
        embedders['OpenAI'] = OpenAIEmbedder(api_key = st.secrets.get("OPENAI_API_KEY", ""))
        models['OpenAI'] = load_model("models/logistic_regression/openai.joblib")
    except Exception as e:
        st.warning(f"OpenAI model not available: {e}")
    
    return embedders, models

embedders, models = load_all_models()

if not models:
    st.error("No models available! Please run the training script first.")
    st.stop()

st.markdown("Enter a news article below. The system will classify it using 4 different embedding models and show a side-by-side comparison.")

article = st.text_area("Paste your article here:", height=200)

if st.button("Classify Article") and article.strip():
    cleaned = clean_text(article)
    
    preds = {}
    probs = {}
    
    for name in models.keys():
        try:
            embedder = embedders[name]
            emb = embedder.encode([cleaned])
            model = models[name]
            prob = model.predict_proba(emb)[0]
            pred = CATEGORY_LABELS[np.argmax(prob)]
            preds[name] = pred
            probs[name] = prob
        except Exception as e:
            st.error(f"Error with {name} model: {e}")
            continue
    
    # Display results
    st.subheader("Predicted Categories:")
    col1, col2 = st.columns(2)
    with col1:
        st.write({name: preds[name] for name in preds})
    with col2:
        st.write({name: [f"{CATEGORY_LABELS[i]}: {probs[name][i]:.2f}" for i in range(len(CATEGORY_LABELS))] for name in probs})
    
    # Bar chart comparison
    st.subheader("Confidence Scores Comparison")
    fig = go.Figure()
    for name in probs:
        fig.add_trace(go.Bar(name=name, x=CATEGORY_LABELS, y=probs[name]))
    fig.update_layout(barmode='group', xaxis_title='Category', yaxis_title='Confidence')
    st.plotly_chart(fig, use_container_width=True)
    
    # Optional: 2D embedding visualization (placeholder)
    # st.subheader("2D Embedding Visualization (PCA/t-SNE)")
    # ...
else:
    st.info("Enter an article and click 'Classify Article' to see results.") 