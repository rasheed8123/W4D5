import pandas as pd
import re
import string
from typing import List

CATEGORY_LABELS = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def preprocess_texts(texts: List[str]) -> List[str]:
    return [clean_text(t) for t in texts]

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[df['category'].isin(CATEGORY_LABELS)]
    df['text'] = preprocess_texts(df['text'].astype(str))
    return df 