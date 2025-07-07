import numpy as np
from typing import List

class GloveEmbedder:
    def __init__(self, glove_path: str):
        self.embeddings = self.load_glove(glove_path)
        self.dim = 300
    def load_glove(self, path):
        embeddings = {}
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings
    def encode(self, texts: List[str]) -> np.ndarray:
        results = []
        for text in texts:
            words = text.split()
            vectors = [self.embeddings[w] for w in words if w in self.embeddings]
            if vectors:
                results.append(np.mean(vectors, axis=0))
            else:
                results.append(np.zeros(self.dim))
        return np.vstack(results)

class BertEmbedder:
    def __init__(self):
        from transformers import BertTokenizer, BertModel
        import torch
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.torch = torch
    def encode(self, texts: List[str]) -> np.ndarray:
        with self.torch.no_grad():
            results = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                results.append(cls_embedding)
            return np.vstack(results)

class SBertEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def encode(self, texts: List[str]) -> np.ndarray:
        return np.vstack(self.model.encode(texts, show_progress_bar=False))

class OpenAIEmbedder:
    def __init__(self, api_key: str):
        import openai
        self.openai = openai
        self.openai.api_key = api_key
        self.model = 'text-embedding-ada-002'
    def encode(self, texts: List[str]) -> np.ndarray:
        results = []
        for text in texts:
            response = self.openai.Embedding.create(input=text, model=self.model)
            results.append(response['data'][0]['embedding'])
        return np.vstack(results) 