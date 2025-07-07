import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def train_classifier(X, y):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    return clf

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    return metrics

def save_metrics(metrics, path):
    import json
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(path):
    import json
    with open(path, 'r') as f:
        return json.load(f) 