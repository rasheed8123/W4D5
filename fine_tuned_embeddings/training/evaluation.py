"""
Evaluation Module for Sales Conversion Prediction
Compares fine-tuned vs generic embeddings and provides comprehensive metrics
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Any
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesConversionEvaluator:
    """Evaluates sales conversion prediction models"""
    
    def __init__(self, generic_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.generic_model_name = generic_model_name
        self.generic_model = None
        self.finetuned_model = None
        self.results = {}
        
    def load_models(self, finetuned_model_path: str):
        """Load both generic and fine-tuned models"""
        logger.info("Loading generic model...")
        self.generic_model = SentenceTransformer(self.generic_model_name)
        
        logger.info("Loading fine-tuned model...")
        self.finetuned_model = SentenceTransformer(finetuned_model_path)
        
        logger.info("Models loaded successfully")
    
    def load_test_data(self, csv_path: str) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
        """Load test data from CSV"""
        df = pd.read_csv(csv_path)
        transcripts = df['call_transcript'].tolist()
        labels = df['conversion_label'].values
        metadata = df.drop(['call_transcript', 'conversion_label'], axis=1)
        
        return transcripts, labels, metadata
    
    def compute_embeddings(self, transcripts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute embeddings using both models"""
        logger.info("Computing generic embeddings...")
        generic_embeddings = self.generic_model.encode(transcripts, show_progress_bar=True)
        
        logger.info("Computing fine-tuned embeddings...")
        finetuned_embeddings = self.finetuned_model.encode(transcripts, show_progress_bar=True)
        
        return generic_embeddings, finetuned_embeddings
    
    def compute_similarity_scores(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute similarity scores to converted vs non-converted prototypes"""
        # Separate embeddings by class
        converted_embeddings = embeddings[labels == 1]
        non_converted_embeddings = embeddings[labels == 0]
        
        if len(converted_embeddings) == 0 or len(non_converted_embeddings) == 0:
            logger.warning("No examples in one of the classes")
            return np.zeros(len(embeddings))
        
        # Compute prototype embeddings
        converted_prototype = np.mean(converted_embeddings, axis=0)
        non_converted_prototype = np.mean(non_converted_embeddings, axis=0)
        
        # Compute similarities
        converted_similarities = np.dot(embeddings, converted_prototype) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(converted_prototype)
        )
        non_converted_similarities = np.dot(embeddings, non_converted_prototype) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(non_converted_prototype)
        )
        
        # Return difference in similarities as conversion score
        return converted_similarities - non_converted_similarities
    
    def evaluate_model(self, embeddings: np.ndarray, labels: np.ndarray, model_name: str) -> Dict[str, float]:
        """Evaluate model performance"""
        # Compute similarity scores
        scores = self.compute_similarity_scores(embeddings, labels)
        
        # Make predictions
        predictions = (scores > 0).astype(int)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1_score': f1_score(labels, predictions, zero_division=0),
            'auc_roc': roc_auc_score(labels, scores),
            'conversion_rate_predicted': np.mean(predictions),
            'conversion_rate_actual': np.mean(labels)
        }
        
        # Store detailed results
        self.results[model_name] = {
            'metrics': metrics,
            'scores': scores,
            'predictions': predictions,
            'embeddings': embeddings
        }
        
        return metrics
    
    def compare_models(self, transcripts: List[str], labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compare generic vs fine-tuned model performance"""
        logger.info("Computing embeddings for both models...")
        generic_embeddings, finetuned_embeddings = self.compute_embeddings(transcripts)
        
        logger.info("Evaluating generic model...")
        generic_metrics = self.evaluate_model(generic_embeddings, labels, 'generic')
        
        logger.info("Evaluating fine-tuned model...")
        finetuned_metrics = self.evaluate_model(finetuned_embeddings, labels, 'finetuned')
        
        # Compute improvement
        improvement = {}
        for metric in generic_metrics.keys():
            if metric in ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score']:
                improvement[metric] = finetuned_metrics[metric] - generic_metrics[metric]
        
        self.results['improvement'] = improvement
        
        return {
            'generic': generic_metrics,
            'finetuned': finetuned_metrics,
            'improvement': improvement
        }
    
    def create_visualizations(self, output_dir: str = 'evaluation_results'):
        """Create comprehensive visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Performance comparison
        self._plot_performance_comparison(output_dir)
        
        # 2. ROC curves
        self._plot_roc_curves(output_dir)
        
        # 3. Precision-Recall curves
        self._plot_precision_recall_curves(output_dir)
        
        # 4. Embedding space visualization
        self._plot_embedding_space(output_dir)
        
        # 5. Score distributions
        self._plot_score_distributions(output_dir)
        
        # 6. Confusion matrices
        self._plot_confusion_matrices(output_dir)
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def _plot_performance_comparison(self, output_dir: str):
        """Plot performance comparison between models"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        generic_scores = [self.results['generic']['metrics'][m] for m in metrics]
        finetuned_scores = [self.results['finetuned']['metrics'][m] for m in metrics]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Generic Model',
            x=metrics,
            y=generic_scores,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Fine-tuned Model',
            x=metrics,
            y=finetuned_scores,
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Metrics',
            yaxis_title='Score',
            barmode='group',
            yaxis=dict(range=[0, 1])
        )
        
        fig.write_html(f"{output_dir}/performance_comparison.html")
        fig.write_image(f"{output_dir}/performance_comparison.png")
    
    def _plot_roc_curves(self, output_dir: str):
        """Plot ROC curves for both models"""
        fig = go.Figure()
        
        # Generic model ROC
        fpr_gen, tpr_gen, _ = roc_curve(
            self.results['generic']['labels'], 
            self.results['generic']['scores']
        )
        auc_gen = self.results['generic']['metrics']['auc_roc']
        
        fig.add_trace(go.Scatter(
            x=fpr_gen, y=tpr_gen,
            name=f'Generic Model (AUC = {auc_gen:.3f})',
            line=dict(color='lightblue')
        ))
        
        # Fine-tuned model ROC
        fpr_ft, tpr_ft, _ = roc_curve(
            self.results['finetuned']['labels'], 
            self.results['finetuned']['scores']
        )
        auc_ft = self.results['finetuned']['metrics']['auc_roc']
        
        fig.add_trace(go.Scatter(
            x=fpr_ft, y=tpr_ft,
            name=f'Fine-tuned Model (AUC = {auc_ft:.3f})',
            line=dict(color='darkblue')
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        fig.write_html(f"{output_dir}/roc_curves.html")
        fig.write_image(f"{output_dir}/roc_curves.png")
    
    def _plot_precision_recall_curves(self, output_dir: str):
        """Plot Precision-Recall curves for both models"""
        fig = go.Figure()
        
        # Generic model PR
        precision_gen, recall_gen, _ = precision_recall_curve(
            self.results['generic']['labels'], 
            self.results['generic']['scores']
        )
        
        fig.add_trace(go.Scatter(
            x=recall_gen, y=precision_gen,
            name='Generic Model',
            line=dict(color='lightblue')
        ))
        
        # Fine-tuned model PR
        precision_ft, recall_ft, _ = precision_recall_curve(
            self.results['finetuned']['labels'], 
            self.results['finetuned']['scores']
        )
        
        fig.add_trace(go.Scatter(
            x=recall_ft, y=precision_ft,
            name='Fine-tuned Model',
            line=dict(color='darkblue')
        ))
        
        fig.update_layout(
            title='Precision-Recall Curves Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        fig.write_html(f"{output_dir}/precision_recall_curves.html")
        fig.write_image(f"{output_dir}/precision_recall_curves.png")
    
    def _plot_embedding_space(self, output_dir: str):
        """Visualize embedding spaces using UMAP"""
        # Reduce dimensionality for visualization
        reducer = umap.UMAP(n_components=2, random_state=42)
        
        # Generic embeddings
        generic_2d = reducer.fit_transform(self.results['generic']['embeddings'])
        
        # Fine-tuned embeddings
        finetuned_2d = reducer.fit_transform(self.results['finetuned']['embeddings'])
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Generic Embeddings', 'Fine-tuned Embeddings')
        )
        
        # Generic embeddings plot
        labels = self.results['generic']['labels']
        fig.add_trace(
            go.Scatter(
                x=generic_2d[labels == 0, 0],
                y=generic_2d[labels == 0, 1],
                mode='markers',
                name='Non-converted (Generic)',
                marker=dict(color='red', size=8),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=generic_2d[labels == 1, 0],
                y=generic_2d[labels == 1, 1],
                mode='markers',
                name='Converted (Generic)',
                marker=dict(color='green', size=8),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Fine-tuned embeddings plot
        fig.add_trace(
            go.Scatter(
                x=finetuned_2d[labels == 0, 0],
                y=finetuned_2d[labels == 0, 1],
                mode='markers',
                name='Non-converted (Fine-tuned)',
                marker=dict(color='red', size=8)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=finetuned_2d[labels == 1, 0],
                y=finetuned_2d[labels == 1, 1],
                mode='markers',
                name='Converted (Fine-tuned)',
                marker=dict(color='green', size=8)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Embedding Space Visualization (UMAP)',
            width=1200,
            height=500
        )
        
        fig.write_html(f"{output_dir}/embedding_space.html")
        fig.write_image(f"{output_dir}/embedding_space.png")
    
    def _plot_score_distributions(self, output_dir: str):
        """Plot score distributions for both models"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Generic Model Scores', 'Fine-tuned Model Scores')
        )
        
        # Generic scores
        labels = self.results['generic']['labels']
        generic_scores = self.results['generic']['scores']
        
        fig.add_trace(
            go.Histogram(
                x=generic_scores[labels == 0],
                name='Non-converted',
                marker_color='red',
                opacity=0.7,
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=generic_scores[labels == 1],
                name='Converted',
                marker_color='green',
                opacity=0.7,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Fine-tuned scores
        finetuned_scores = self.results['finetuned']['scores']
        
        fig.add_trace(
            go.Histogram(
                x=finetuned_scores[labels == 0],
                name='Non-converted',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=finetuned_scores[labels == 1],
                name='Converted',
                marker_color='green',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Score Distributions by Conversion Status',
            width=1200,
            height=500
        )
        
        fig.write_html(f"{output_dir}/score_distributions.html")
        fig.write_image(f"{output_dir}/score_distributions.png")
    
    def _plot_confusion_matrices(self, output_dir: str):
        """Plot confusion matrices for both models"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Generic Model', 'Fine-tuned Model'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # Generic confusion matrix
        cm_gen = confusion_matrix(
            self.results['generic']['labels'],
            self.results['generic']['predictions']
        )
        
        fig.add_trace(
            go.Heatmap(
                z=cm_gen,
                x=['Predicted Non-converted', 'Predicted Converted'],
                y=['Actual Non-converted', 'Actual Converted'],
                text=cm_gen,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Blues',
                showscale=False
            ),
            row=1, col=1
        )
        
        # Fine-tuned confusion matrix
        cm_ft = confusion_matrix(
            self.results['finetuned']['labels'],
            self.results['finetuned']['predictions']
        )
        
        fig.add_trace(
            go.Heatmap(
                z=cm_ft,
                x=['Predicted Non-converted', 'Predicted Converted'],
                y=['Actual Non-converted', 'Actual Converted'],
                text=cm_ft,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Blues',
                showscale=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Confusion Matrices',
            width=1000,
            height=400
        )
        
        fig.write_html(f"{output_dir}/confusion_matrices.html")
        fig.write_image(f"{output_dir}/confusion_matrices.png")
    
    def generate_report(self, output_dir: str = 'evaluation_results') -> str:
        """Generate comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations
        self.create_visualizations(output_dir)
        
        # Generate text report
        report_path = f"{output_dir}/evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("SALES CONVERSION PREDICTION EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model comparison
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("-" * 30 + "\n")
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
            f.write(f"{'Metric':<15} {'Generic':<10} {'Fine-tuned':<12} {'Improvement':<12}\n")
            f.write("-" * 50 + "\n")
            
            for metric in metrics:
                generic_val = self.results['generic']['metrics'][metric]
                finetuned_val = self.results['finetuned']['metrics'][metric]
                improvement = self.results['improvement'][metric]
                
                f.write(f"{metric:<15} {generic_val:<10.4f} {finetuned_val:<12.4f} {improvement:<12.4f}\n")
            
            f.write("\n")
            
            # Detailed metrics
            f.write("DETAILED METRICS\n")
            f.write("-" * 20 + "\n")
            
            for model_name in ['generic', 'finetuned']:
                f.write(f"\n{model_name.upper()} MODEL:\n")
                for metric, value in self.results[model_name]['metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\n")
            
            # Improvement analysis
            f.write("IMPROVEMENT ANALYSIS\n")
            f.write("-" * 20 + "\n")
            
            total_improvement = sum(self.results['improvement'].values())
            f.write(f"Total improvement across all metrics: {total_improvement:.4f}\n")
            
            best_improvement = max(self.results['improvement'].items(), key=lambda x: x[1])
            f.write(f"Best improvement: {best_improvement[0]} (+{best_improvement[1]:.4f})\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            if self.results['improvement']['auc_roc'] > 0.05:
                f.write("✓ Fine-tuned model shows significant improvement in AUC-ROC\n")
            else:
                f.write("⚠ Fine-tuned model shows minimal improvement in AUC-ROC\n")
            
            if self.results['improvement']['f1_score'] > 0.05:
                f.write("✓ Fine-tuned model shows significant improvement in F1-score\n")
            else:
                f.write("⚠ Fine-tuned model shows minimal improvement in F1-score\n")
            
            f.write("\n")
            f.write("Visualization files generated:\n")
            f.write("- performance_comparison.html/png\n")
            f.write("- roc_curves.html/png\n")
            f.write("- precision_recall_curves.html/png\n")
            f.write("- embedding_space.html/png\n")
            f.write("- score_distributions.html/png\n")
            f.write("- confusion_matrices.html/png\n")
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path

def main():
    """Main evaluation script"""
    # Initialize evaluator
    evaluator = SalesConversionEvaluator()
    
    # Load models
    evaluator.load_models('models/finetuned_sales_model')
    
    # Load test data
    transcripts, labels, metadata = evaluator.load_test_data('data/sample_transcripts.csv')
    
    # Store labels for later use
    evaluator.results['generic']['labels'] = labels
    evaluator.results['finetuned']['labels'] = labels
    
    # Compare models
    results = evaluator.compare_models(transcripts, labels)
    
    # Print results
    print("\nMODEL COMPARISON RESULTS:")
    print("=" * 40)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    print(f"{'Metric':<15} {'Generic':<10} {'Fine-tuned':<12} {'Improvement':<12}")
    print("-" * 50)
    
    for metric in metrics:
        generic_val = results['generic'][metric]
        finetuned_val = results['finetuned'][metric]
        improvement = results['improvement'][metric]
        
        print(f"{metric:<15} {generic_val:<10.4f} {finetuned_val:<12.4f} {improvement:<12.4f}")
    
    # Generate comprehensive report
    report_path = evaluator.generate_report()
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main() 