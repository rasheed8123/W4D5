"""
Streamlit UI for Sales Conversion AI System
Provides interactive interface for conversion prediction and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.embedder import SalesEmbedder, EmbeddingComparison
from pipeline.scorer import ConversionPredictor, HybridScorer
from pipeline.langchain_chain import SalesConversionChain
from training.evaluation import SalesConversionEvaluator

# Page configuration
st.set_page_config(
    page_title="Sales Conversion AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        color: #28a745;
        font-weight: bold;
    }
    .prediction-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .prediction-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class SalesConversionApp:
    """Main Streamlit application for sales conversion AI"""
    
    def __init__(self):
        self.embedder = None
        self.predictor = None
        self.pipeline = None
        self.evaluator = None
        
        # Initialize session state
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
    
    def run(self):
        """Run the main application"""
        # Header
        st.markdown('<h1 class="main-header">🎯 Sales Conversion AI</h1>', unsafe_allow_html=True)
        st.markdown("### Intelligent Sales Call Analysis & Conversion Prediction")
        
        # Sidebar
        self._create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📞 Call Analysis", 
            "📊 Performance Dashboard", 
            "🔍 Model Comparison", 
            "📈 Training & Evaluation",
            "⚙️ Settings"
        ])
        
        with tab1:
            self._call_analysis_tab()
        
        with tab2:
            self._performance_dashboard_tab()
        
        with tab3:
            self._model_comparison_tab()
        
        with tab4:
            self._training_evaluation_tab()
        
        with tab5:
            self._settings_tab()
    
    def _create_sidebar(self):
        """Create sidebar with model selection and upload options"""
        st.sidebar.title("🔧 Configuration")
        
        # Model selection
        st.sidebar.subheader("Model Selection")
        model_option = st.sidebar.selectbox(
            "Choose Model",
            ["Fine-tuned Model", "Generic Model", "Both (Comparison)"],
            help="Select which model(s) to use for analysis"
        )
        
        # Model path input
        if model_option in ["Fine-tuned Model", "Both (Comparison)"]:
            model_path = st.sidebar.text_input(
                "Fine-tuned Model Path",
                value="models/finetuned_sales_model",
                help="Path to the fine-tuned model directory"
            )
        else:
            model_path = None
        
        # Load models button
        if st.sidebar.button("🚀 Load Models", type="primary"):
            with st.spinner("Loading models..."):
                self._load_models(model_path, model_option)
        
        # Data upload
        st.sidebar.subheader("📁 Data Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Sales Call Data",
            type=['csv'],
            help="Upload CSV file with call transcripts and conversion labels"
        )
        
        if uploaded_file is not None:
            self._load_uploaded_data(uploaded_file)
        
        # System info
        st.sidebar.subheader("ℹ️ System Info")
        if self.embedder:
            info = self.embedder.get_model_info()
            st.sidebar.write(f"**Model:** {'Fine-tuned' if info['is_finetuned'] else 'Generic'}")
            st.sidebar.write(f"**Embedding Dim:** {info['embedding_dimension']}")
            st.sidebar.write(f"**Prototypes:** {'✓' if info['prototypes_computed'] else '✗'}")
    
    def _load_models(self, model_path: str, model_option: str):
        """Load the selected models"""
        try:
            if model_option == "Fine-tuned Model":
                self.embedder = SalesEmbedder(model_path=model_path)
                st.success("✅ Fine-tuned model loaded successfully!")
                
            elif model_option == "Generic Model":
                self.embedder = SalesEmbedder()
                st.success("✅ Generic model loaded successfully!")
                
            elif model_option == "Both (Comparison)":
                self.evaluator = SalesConversionEvaluator()
                self.evaluator.load_models(model_path)
                st.success("✅ Both models loaded for comparison!")
            
            # Load training data if available
            if os.path.exists('data/sample_transcripts.csv'):
                self._load_training_data()
                
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
    
    def _load_training_data(self):
        """Load training data and initialize predictor"""
        try:
            df = pd.read_csv('data/sample_transcripts.csv')
            transcripts = df['call_transcript'].tolist()
            labels = df['conversion_label'].values
            
            if self.embedder:
                # Compute prototypes
                self.embedder.compute_prototypes(transcripts, labels)
                
                # Initialize predictor
                embeddings = self.embedder.encode(transcripts)
                self.predictor = ConversionPredictor(
                    embeddings=embeddings,
                    labels=labels,
                    transcripts=transcripts,
                    scorer_type='hybrid'
                )
                
                # Initialize pipeline
                self.pipeline = SalesConversionChain()
                self.pipeline.load_training_data('data/sample_transcripts.csv')
                
                st.success("✅ Training data loaded and predictor initialized!")
                
        except Exception as e:
            st.warning(f"⚠️ Could not load training data: {str(e)}")
    
    def _load_uploaded_data(self, uploaded_file):
        """Load uploaded data"""
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            st.success(f"✅ Uploaded {len(df)} records successfully!")
        except Exception as e:
            st.error(f"❌ Error loading uploaded file: {str(e)}")
    
    def _call_analysis_tab(self):
        """Call analysis tab"""
        st.header("📞 Sales Call Analysis")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter Call Transcript")
            transcript_input = st.text_area(
                "Paste the sales call transcript here:",
                height=200,
                placeholder="Agent: Hi, I'm calling about your recent inquiry...\nCustomer: Yes, we're interested...\nAgent: Great! Let me tell you about our features..."
            )
        
        with col2:
            st.subheader("Analysis Options")
            include_analysis = st.checkbox("Include Detailed Analysis", value=True)
            show_similar_cases = st.checkbox("Show Similar Cases", value=True)
            top_k_similar = st.slider("Number of Similar Cases", 1, 10, 3)
        
        # Analysis button
        if st.button("🔍 Analyze Call", type="primary", disabled=not transcript_input.strip()):
            if not self.predictor:
                st.error("❌ Please load models and training data first!")
                return
            
            with st.spinner("Analyzing call transcript..."):
                self._analyze_call(transcript_input, include_analysis, show_similar_cases, top_k_similar)
    
    def _analyze_call(self, transcript: str, include_analysis: bool, 
                     show_similar_cases: bool, top_k: int):
        """Analyze a single call"""
        try:
            # Get prediction
            embedding = self.embedder.encode(transcript)
            prediction = self.predictor.predict_conversion(embedding)
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Conversion Score",
                    f"{prediction.conversion_score:.3f}",
                    delta=f"{prediction.conversion_score - 0:.3f}"
                )
            
            with col2:
                probability = prediction.conversion_probability
                if probability > 0.7:
                    prob_class = "prediction-high"
                elif probability > 0.4:
                    prob_class = "prediction-medium"
                else:
                    prob_class = "prediction-low"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Conversion Probability</h4>
                    <p class="{prob_class}">{probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.metric(
                    "Confidence",
                    f"{prediction.confidence:.1%}"
                )
            
            with col4:
                prediction_text = "✅ Convert" if prediction.prediction else "❌ No Convert"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Prediction</h4>
                    <p class="{prob_class}">{prediction_text}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed analysis
            if include_analysis:
                st.subheader("📋 Detailed Analysis")
                st.write(prediction.reasoning)
                
                if hasattr(prediction, 'detailed_analysis') and prediction.detailed_analysis:
                    analysis = prediction.detailed_analysis
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Key Factors:**")
                        for factor in analysis.get('key_factors', []):
                            st.write(f"• {factor}")
                    
                    with col2:
                        st.write("**Recommendations:**")
                        for rec in analysis.get('recommendations', []):
                            st.write(f"• {rec}")
            
            # Similar cases
            if show_similar_cases and prediction.similar_cases:
                st.subheader("🔍 Similar Historical Cases")
                
                for i, case in enumerate(prediction.similar_cases[:top_k]):
                    with st.expander(f"Case {i+1} (Similarity: {case['similarity']:.3f})"):
                        st.write(case['transcript'])
                        st.write(f"**Actual Conversion:** {'✅ Yes' if case['actual_conversion'] else '❌ No'}")
            
            # Store prediction
            st.session_state.predictions.append({
                'transcript': transcript,
                'prediction': prediction,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            st.error(f"❌ Error analyzing call: {str(e)}")
    
    def _performance_dashboard_tab(self):
        """Performance dashboard tab"""
        st.header("📊 Performance Dashboard")
        
        if not st.session_state.predictions:
            st.info("📝 No predictions made yet. Analyze some calls first!")
            return
        
        # Convert predictions to DataFrame
        df_predictions = pd.DataFrame([
            {
                'timestamp': pred['timestamp'],
                'conversion_score': pred['prediction'].conversion_score,
                'conversion_probability': pred['prediction'].conversion_probability,
                'confidence': pred['prediction'].confidence,
                'prediction': pred['prediction'].prediction
            }
            for pred in st.session_state.predictions
        ])
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Calls Analyzed", len(df_predictions))
        
        with col2:
            avg_prob = df_predictions['conversion_probability'].mean()
            st.metric("Average Conversion Probability", f"{avg_prob:.1%}")
        
        with col3:
            avg_confidence = df_predictions['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        with col4:
            predicted_conversions = (df_predictions['prediction'] == 1).sum()
            st.metric("Predicted Conversions", predicted_conversions)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Conversion probability distribution
            fig_prob = px.histogram(
                df_predictions,
                x='conversion_probability',
                nbins=20,
                title="Conversion Probability Distribution",
                labels={'conversion_probability': 'Probability', 'count': 'Number of Calls'}
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with col2:
            # Confidence vs probability scatter
            fig_scatter = px.scatter(
                df_predictions,
                x='conversion_probability',
                y='confidence',
                title="Confidence vs Conversion Probability",
                labels={'conversion_probability': 'Conversion Probability', 'confidence': 'Confidence'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Time series
        if len(df_predictions) > 1:
            st.subheader("📈 Trends Over Time")
            
            fig_trend = px.line(
                df_predictions,
                x='timestamp',
                y='conversion_probability',
                title="Conversion Probability Trends",
                labels={'timestamp': 'Time', 'conversion_probability': 'Conversion Probability'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    
    def _model_comparison_tab(self):
        """Model comparison tab"""
        st.header("🔍 Model Comparison")
        
        if not self.evaluator:
            st.info("📝 Load both models to enable comparison!")
            return
        
        # Load test data
        if st.button("🔄 Run Model Comparison"):
            with st.spinner("Running model comparison..."):
                self._run_model_comparison()
    
    def _run_model_comparison(self):
        """Run model comparison analysis"""
        try:
            # Load test data
            df = pd.read_csv('data/sample_transcripts.csv')
            transcripts = df['call_transcript'].tolist()
            labels = df['conversion_label'].values
            
            # Compare models
            results = self.evaluator.compare_models(transcripts, labels)
            
            # Display results
            st.subheader("📊 Model Performance Comparison")
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
            
            # Create comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Generic Model',
                x=metrics,
                y=[results['generic'][m] for m in metrics],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Fine-tuned Model',
                x=metrics,
                y=[results['finetuned'][m] for m in metrics],
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Metrics',
                yaxis_title='Score',
                barmode='group',
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Improvement metrics
            st.subheader("📈 Improvement Analysis")
            
            improvement_df = pd.DataFrame([
                {
                    'Metric': metric,
                    'Generic': results['generic'][metric],
                    'Fine-tuned': results['finetuned'][metric],
                    'Improvement': results['improvement'][metric]
                }
                for metric in metrics
            ])
            
            st.dataframe(improvement_df, use_container_width=True)
            
            # Generate visualizations
            if st.button("📊 Generate Detailed Visualizations"):
                with st.spinner("Generating visualizations..."):
                    self.evaluator.create_visualizations('evaluation_results')
                    st.success("✅ Visualizations saved to 'evaluation_results' folder!")
            
        except Exception as e:
            st.error(f"❌ Error in model comparison: {str(e)}")
    
    def _training_evaluation_tab(self):
        """Training and evaluation tab"""
        st.header("📈 Training & Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Training Pipeline")
            
            if st.button("🔄 Create Contrastive Dataset"):
                with st.spinner("Creating contrastive dataset..."):
                    self._create_contrastive_dataset()
            
            if st.button("🚀 Train Fine-tuned Model"):
                with st.spinner("Training model..."):
                    self._train_model()
        
        with col2:
            st.subheader("📊 Evaluation")
            
            if st.button("📈 Run Evaluation"):
                with st.spinner("Running evaluation..."):
                    self._run_evaluation()
    
    def _create_contrastive_dataset(self):
        """Create contrastive dataset"""
        try:
            from training.contrastive_dataset import ContrastiveDatasetCreator
            
            creator = ContrastiveDatasetCreator()
            creator.load_transcripts('data/sample_transcripts.csv')
            
            # Create triplets
            triplets = creator.create_triplets(n_triplets=100)
            hard_triplets = creator.create_hard_negative_mining_triplets(n_triplets=50)
            
            # Combine and split
            all_triplets = triplets + hard_triplets
            train_triplets, val_triplets = creator.create_train_val_split(all_triplets)
            
            # Save datasets
            creator.save_triplets(train_triplets, 'data/train_triplets.json')
            creator.save_triplets(val_triplets, 'data/val_triplets.json')
            
            st.success(f"✅ Created {len(all_triplets)} contrastive triplets!")
            
        except Exception as e:
            st.error(f"❌ Error creating dataset: {str(e)}")
    
    def _train_model(self):
        """Train the fine-tuned model"""
        try:
            from training.train_finetune import SalesEmbeddingTrainer
            
            trainer = SalesEmbeddingTrainer()
            
            # Load data
            train_triplets, val_triplets = trainer.load_data(
                'data/train_triplets.json',
                'data/val_triplets.json'
            )
            
            # Train model
            trainer.train_with_sentence_transformers(
                train_triplets=train_triplets,
                val_triplets=val_triplets,
                output_path='models/finetuned_sales_model',
                epochs=3,  # Reduced for demo
                batch_size=16,
                learning_rate=2e-5
            )
            
            st.success("✅ Model training completed!")
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
    
    def _run_evaluation(self):
        """Run comprehensive evaluation"""
        try:
            evaluator = SalesConversionEvaluator()
            evaluator.load_models('models/finetuned_sales_model')
            
            # Load test data
            df = pd.read_csv('data/sample_transcripts.csv')
            transcripts = df['call_transcript'].tolist()
            labels = df['conversion_label'].values
            
            # Run evaluation
            results = evaluator.compare_models(transcripts, labels)
            
            # Generate report
            report_path = evaluator.generate_report()
            
            st.success(f"✅ Evaluation completed! Report saved to: {report_path}")
            
            # Display summary
            st.subheader("📊 Evaluation Summary")
            
            improvement = results['improvement']
            total_improvement = sum(improvement.values())
            
            st.metric("Total Improvement", f"{total_improvement:.3f}")
            
            best_metric = max(improvement.items(), key=lambda x: x[1])
            st.metric(f"Best Improvement ({best_metric[0]})", f"{best_metric[1]:.3f}")
            
        except Exception as e:
            st.error(f"❌ Error in evaluation: {str(e)}")
    
    def _settings_tab(self):
        """Settings tab"""
        st.header("⚙️ Settings")
        
        st.subheader("🔧 Model Configuration")
        
        # Model parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Embedding Model:**")
            st.write("- Generic: sentence-transformers/all-MiniLM-L6-v2")
            st.write("- Fine-tuned: Custom model trained on sales data")
            
            st.write("**Scoring Methods:**")
            st.write("- Cosine Similarity")
            st.write("- Euclidean Distance")
            st.write("- MMR (Maximum Marginal Relevance)")
            st.write("- Hybrid (Combined approach)")
        
        with col2:
            st.write("**Training Parameters:**")
            st.write("- Loss: Contrastive Loss")
            st.write("- Optimizer: AdamW")
            st.write("- Learning Rate: 2e-5")
            st.write("- Batch Size: 16")
            st.write("- Epochs: 10")
        
        st.subheader("📁 File Structure")
        st.code("""
sales_conversion_ai/
├── data/
│   ├── sample_transcripts.csv
│   ├── train_triplets.json
│   └── val_triplets.json
├── models/
│   └── finetuned_sales_model/
├── training/
│   ├── contrastive_dataset.py
│   ├── train_finetune.py
│   └── evaluation.py
├── pipeline/
│   ├── embedder.py
│   ├── scorer.py
│   └── langchain_chain.py
└── ui/
    └── app.py
        """)
        
        st.subheader("🔄 Reset Application")
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.predictions = []
            st.session_state.uploaded_data = None
            st.success("✅ Application data cleared!")

def main():
    """Main function to run the Streamlit app"""
    app = SalesConversionApp()
    app.run()

if __name__ == "__main__":
    main() 