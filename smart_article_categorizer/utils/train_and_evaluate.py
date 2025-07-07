import os
import numpy as np
import sys
from preprocess import load_data, CATEGORY_LABELS
from embedding_models import GloveEmbedder, BertEmbedder, SBertEmbedder, OpenAIEmbedder
from model_utils import train_classifier, save_model, evaluate_classifier
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import json

# Load env variables
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Paths
DATA_PATH = '../data/articles.csv'
GLOVE_PATH = '../embeddings/glove.6B.300d.txt'
MODEL_DIR = '../models/logistic_regression/'
METRICS_DIR = '../models/logistic_regression/'

# Helper: Save metrics with JSON-safe types
def save_metrics(metrics, filepath):
    # Convert all NumPy types to native Python types
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        return o

    with open(filepath, 'w') as f:
        json.dump(metrics, f, default=convert, indent=2)


# Load data
print(f"Loading data from {DATA_PATH}...")
try:
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} articles")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    print("Please ensure articles.csv exists in the data/ directory")
    sys.exit(1)

X = df['text'].tolist()
y = df['category'].tolist()

# Train/test split
if len(X) < 30:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize embedders
print("Initializing embedding models...")
embedders = {}

try:
    glove = GloveEmbedder(GLOVE_PATH)
    print("✓ GloVe embedder initialized")
    embedders['glove'] = glove
except FileNotFoundError:
    print(f"✗ GloVe file not found at {GLOVE_PATH}, skipping...")

try:
    bert = BertEmbedder()
    print("✓ BERT embedder initialized")
    embedders['bert'] = bert
except Exception as e:
    print(f"✗ BERT embedder error: {e}")

try:
    sbert = SBertEmbedder()
    print("✓ SBERT embedder initialized")
    embedders['sbert'] = sbert
except Exception as e:
    print(f"✗ SBERT embedder error: {e}")

try:
    openai = OpenAIEmbedder(api_key=OPENAI_API_KEY)
    print("✓ OpenAI embedder initialized")
    embedders['openai'] = openai
except Exception as e:
    print(f"✗ OpenAI embedder error: {e}")
    print("Hint: Update your embedding_models.py for openai>=1.0")

if not embedders:
    print("❌ No embedding models could be initialized. Exiting.")
    sys.exit(1)

# Training
print(f"\nTraining classifiers for {len(embedders)} embedding models...")

for name, embedder in embedders.items():
    try:
        print(f'\n--- Processing {name} embeddings ---')
        X_train_emb = embedder.encode(X_train)
        X_test_emb = embedder.encode(X_test)

        print(f'Training {name} classifier...')
        clf = train_classifier(X_train_emb, y_train)
        save_model(clf, os.path.join(MODEL_DIR, f'{name}.joblib'))

        print(f'Evaluating {name} classifier...')
        metrics = evaluate_classifier(clf, X_test_emb, y_test)
        save_metrics(metrics, os.path.join(METRICS_DIR, f'{name}_metrics.json'))

        print(f'✓ {name} completed!')
        print(f'  Accuracy: {metrics["accuracy"]:.3f}')
        print(f'  F1-Score: {metrics["f1"]:.3f}')
    except Exception as e:
        print(f'✗ Error processing {name}: {e}')
        continue

print("\n✅ Training completed!")
