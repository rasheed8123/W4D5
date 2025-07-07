import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from dataclasses import dataclass

@dataclass
class ClassificationResult:
    """Result of document classification."""
    doc_type: str
    confidence: float
    structure_tags: List[str]
    features: Dict[str, float]

class DocumentClassifier:
    """Classifies documents into different types for adaptive chunking."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.doc_types = [
            'technical_docs',
            'api_references', 
            'troubleshooting_tickets',
            'policy_documents',
            'code_tutorials',
            'general_docs'
        ]
        
        self.structure_tags = [
            'has_headers', 'has_code_blocks', 'has_tables', 'has_lists',
            'has_api_endpoints', 'has_error_messages', 'has_steps',
            'has_policies', 'has_clauses', 'has_examples'
        ]
        
        # Rule-based patterns
        self.patterns = {
            'technical_docs': [
                r'\b(installation|setup|configuration|deployment|architecture)\b',
                r'\b(requirements|prerequisites|dependencies)\b',
                r'\b(technical|system|infrastructure)\b'
            ],
            'api_references': [
                r'\b(api|endpoint|method|function|parameter|response)\b',
                r'\b(get|post|put|delete|patch)\b',
                r'\b(http|https|rest|graphql)\b',
                r'\b(status code|error code|response code)\b'
            ],
            'troubleshooting_tickets': [
                r'\b(error|issue|problem|bug|failure|crash)\b',
                r'\b(troubleshoot|debug|fix|resolve|solution)\b',
                r'\b(log|stack trace|exception|error message)\b',
                r'\b(case|ticket|incident|support)\b'
            ],
            'policy_documents': [
                r'\b(policy|procedure|guideline|rule|regulation)\b',
                r'\b(compliance|audit|security|privacy)\b',
                r'\b(terms|conditions|agreement|contract)\b',
                r'\b(shall|must|should|will|may)\b'
            ],
            'code_tutorials': [
                r'\b(tutorial|guide|example|sample|demo)\b',
                r'\b(step|instruction|how to|walkthrough)\b',
                r'```[\w]*\n',  # Code blocks
                r'\b(import|from|class|def|function)\b'
            ]
        }
        
        # Initialize ML model
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def classify_document(self, content: str, metadata: Optional[Dict] = None) -> ClassificationResult:
        """
        Classify a document using both rule-based and ML approaches.
        
        Args:
            content: Document content
            metadata: Optional document metadata
            
        Returns:
            ClassificationResult with type, confidence, and structure tags
        """
        # Extract features
        features = self._extract_features(content, metadata)
        
        # Rule-based classification
        rule_result = self._rule_based_classification(content, features)
        
        # ML-based classification (if model is trained)
        ml_result = None
        if self.is_trained:
            ml_result = self._ml_classification(content)
        
        # Combine results
        final_result = self._combine_classifications(rule_result, ml_result, features)
        
        return final_result
    
    def _extract_features(self, content: str, metadata: Optional[Dict] = None) -> Dict[str, float]:
        """Extract features from document content."""
        features = {}
        
        # Basic text features
        features['word_count'] = len(content.split())
        features['char_count'] = len(content)
        features['avg_word_length'] = np.mean([len(word) for word in content.split()]) if content.split() else 0
        
        # Structure features
        features['has_headers'] = 1.0 if re.search(r'^#{1,6}\s', content, re.MULTILINE) else 0.0
        features['has_code_blocks'] = 1.0 if re.search(r'```[\w]*\n', content) else 0.0
        features['has_tables'] = 1.0 if re.search(r'\|.*\|', content) or '--- TABLE ---' in content else 0.0
        features['has_lists'] = 1.0 if re.search(r'^[\*\-]\s|^\d+\.\s', content, re.MULTILINE) else 0.0
        
        # API-specific features
        features['has_api_endpoints'] = 1.0 if re.search(r'/(api|v\d+)/', content) else 0.0
        features['has_http_methods'] = 1.0 if re.search(r'\b(GET|POST|PUT|DELETE|PATCH)\b', content) else 0.0
        
        # Error/troubleshooting features
        features['has_error_messages'] = 1.0 if re.search(r'\b(error|exception|failure|crash)\b', content.lower()) else 0.0
        features['has_steps'] = 1.0 if re.search(r'\b(step \d+|first|second|finally)\b', content.lower()) else 0.0
        
        # Policy features
        features['has_policies'] = 1.0 if re.search(r'\b(policy|procedure|guideline)\b', content.lower()) else 0.0
        features['has_clauses'] = 1.0 if re.search(r'\b(clause|section|article)\b', content.lower()) else 0.0
        
        # Tutorial features
        features['has_examples'] = 1.0 if re.search(r'\b(example|sample|demo)\b', content.lower()) else 0.0
        features['has_imports'] = 1.0 if re.search(r'\b(import|from)\b', content) else 0.0
        
        # File type features
        if metadata and 'file_type' in metadata:
            features['is_markdown'] = 1.0 if metadata['file_type'] == '.md' else 0.0
            features['is_pdf'] = 1.0 if metadata['file_type'] == '.pdf' else 0.0
            features['is_docx'] = 1.0 if metadata['file_type'] in ['.docx', '.doc'] else 0.0
        
        return features
    
    def _rule_based_classification(self, content: str, features: Dict[str, float]) -> Tuple[str, float]:
        """Perform rule-based classification."""
        scores = {doc_type: 0.0 for doc_type in self.doc_types}
        
        content_lower = content.lower()
        
        for doc_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                scores[doc_type] += matches
        
        # Add feature-based scoring
        if features['has_api_endpoints'] > 0:
            scores['api_references'] += 2.0
        if features['has_error_messages'] > 0:
            scores['troubleshooting_tickets'] += 2.0
        if features['has_policies'] > 0:
            scores['policy_documents'] += 2.0
        if features['has_code_blocks'] > 0:
            scores['code_tutorials'] += 1.5
        if features['has_headers'] > 0:
            scores['technical_docs'] += 1.0
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        
        # Get best match
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, confidence
    
    def _ml_classification(self, content: str) -> Tuple[str, float]:
        """Perform ML-based classification."""
        try:
            # Transform content
            X = self.vectorizer.transform([content])
            
            # Predict
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            
            # Get confidence
            confidence = max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"ML classification error: {e}")
            return 'general_docs', 0.5
    
    def _combine_classifications(self, rule_result: Tuple[str, float], 
                               ml_result: Optional[Tuple[str, float]], 
                               features: Dict[str, float]) -> ClassificationResult:
        """Combine rule-based and ML classification results."""
        
        rule_type, rule_confidence = rule_result
        
        if ml_result is None:
            # Use only rule-based result
            final_type = rule_type
            final_confidence = rule_confidence
        else:
            ml_type, ml_confidence = ml_result
            
            # Weighted combination (favor ML if confidence is high)
            if ml_confidence > 0.7:
                final_type = ml_type
                final_confidence = ml_confidence
            elif rule_confidence > 0.6:
                final_type = rule_type
                final_confidence = rule_confidence
            else:
                # Use ML result with lower confidence
                final_type = ml_type
                final_confidence = ml_confidence
        
        # Extract structure tags
        structure_tags = []
        for tag in self.structure_tags:
            if features.get(tag, 0) > 0:
                structure_tags.append(tag)
        
        return ClassificationResult(
            doc_type=final_type,
            confidence=final_confidence,
            structure_tags=structure_tags,
            features=features
        )
    
    def train_model(self, training_data: List[Tuple[str, str]]):
        """
        Train the ML classifier with labeled data.
        
        Args:
            training_data: List of (content, label) tuples
        """
        if not training_data:
            print("No training data provided")
            return
        
        # Prepare data
        contents, labels = zip(*training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            contents, labels, test_size=0.2, random_state=42
        )
        
        # Transform text
        X_train_transformed = self.vectorizer.fit_transform(X_train)
        X_test_transformed = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_transformed, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
    
    def save_model(self, model_path: str):
        """Save the trained model."""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {model_path}")
    
    def get_classification_stats(self) -> Dict:
        """Get statistics about the classifier."""
        return {
            'doc_types': self.doc_types,
            'structure_tags': self.structure_tags,
            'is_trained': self.is_trained,
            'patterns_count': {k: len(v) for k, v in self.patterns.items()}
        } 