import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import tempfile
from typing import List, Dict, Tuple

# Import our custom modules
from utils.document_processor import DocumentProcessor
from utils.embedding_manager import EmbeddingManager
from utils.evaluation_metrics import EvaluationMetrics
from data.sample_legal_documents import get_sample_documents, get_sample_queries

# Page configuration
st.set_page_config(
    page_title="Indian Legal Document Retrieval System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .method-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-highlight {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        text-align: center;
        font-weight: bold;
    }
    .result-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = EvaluationMetrics()
if 'search_results' not in st.session_state:
    st.session_state.search_results = {}

def initialize_system():
    """Initialize the embedding manager with sample documents."""
    if st.session_state.embedding_manager is None:
        with st.spinner("Initializing the legal document retrieval system..."):
            # Create embedding manager
            st.session_state.embedding_manager = EmbeddingManager()
            
            # Load sample documents
            sample_docs = get_sample_documents()
            st.session_state.documents = sample_docs
            
            # Add documents to embedding manager
            st.session_state.embedding_manager.add_documents(sample_docs)
            
            st.success(f"System initialized with {len(sample_docs)} legal documents!")

def perform_search(query: str) -> Dict[str, List[Tuple[int, float, Dict]]]:
    """Perform search using all four similarity methods."""
    if st.session_state.embedding_manager is None:
        st.error("Please initialize the system first!")
        return {}
    
    results = {}
    
    # Cosine Similarity
    try:
        results['Cosine Similarity'] = st.session_state.embedding_manager.cosine_similarity_search(query, k=5)
    except Exception as e:
        st.error(f"Error in Cosine Similarity: {e}")
        results['Cosine Similarity'] = []
    
    # Euclidean Distance
    try:
        results['Euclidean Distance'] = st.session_state.embedding_manager.euclidean_distance_search(query, k=5)
    except Exception as e:
        st.error(f"Error in Euclidean Distance: {e}")
        results['Euclidean Distance'] = []
    
    # MMR
    try:
        results['MMR'] = st.session_state.embedding_manager.mmr_search(query, k=5, lambda_param=0.5)
    except Exception as e:
        st.error(f"Error in MMR: {e}")
        results['MMR'] = []
    
    # Hybrid
    try:
        results['Hybrid'] = st.session_state.embedding_manager.hybrid_search(query, k=5, cosine_weight=0.6)
    except Exception as e:
        st.error(f"Error in Hybrid: {e}")
        results['Hybrid'] = []
    
    return results

def display_search_results(results: Dict[str, List[Tuple[int, float, Dict]]]):
    """Display search results in a 4-column layout."""
    if not results:
        st.warning("No search results to display.")
        return
    
    # Create 4 columns for the similarity methods
    cols = st.columns(4)
    
    for i, (method_name, method_results) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"<div class='method-card'><h3>{method_name}</h3></div>", unsafe_allow_html=True)
            
            if not method_results:
                st.info("No results found.")
                continue
            
            for j, (idx, score, doc) in enumerate(method_results):
                with st.expander(f"Result {j+1} (Score: {score:.3f})"):
                    st.markdown(f"**Document:** {doc.get('document_name', 'Unknown')}")
                    st.markdown(f"**Section:** {doc.get('title', 'Unknown')}")
                    st.markdown(f"**Category:** {doc.get('category', 'Unknown')}")
                    
                    # Show snippet of content
                    content = doc.get('content', '')
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    st.markdown(f"**Content:** {snippet}")
                    
                    st.markdown(f"**Similarity Score:** {score:.3f}")

def display_metrics_dashboard(results: Dict[str, List[Tuple[int, float, Dict]]], query: str):
    """Display evaluation metrics dashboard."""
    if not results:
        return
    
    st.markdown("## 📊 Performance Metrics Dashboard")
    
    # Calculate metrics
    evaluation_results = st.session_state.evaluation_metrics.evaluate_search_results(
        query=query,
        results=results,
        all_docs=st.session_state.documents
    )
    
    # Display metrics in a table
    metrics_df = pd.DataFrame(evaluation_results).T
    metrics_df = metrics_df.round(3)
    
    st.markdown("### 📈 Method Performance Comparison")
    st.dataframe(metrics_df, use_container_width=True)
    
    # Create visualizations
    create_metrics_visualizations(evaluation_results)

def create_metrics_visualizations(evaluation_results: Dict[str, Dict]):
    """Create interactive visualizations for metrics."""
    if not evaluation_results:
        return
    
    # Prepare data for plotting
    methods = list(evaluation_results.keys())
    metrics = ['diversity_section', 'diversity_document', 'serendipity', 'avg_similarity']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Section Diversity', 'Document Diversity', 'Serendipity', 'Average Similarity'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        values = [evaluation_results[method].get(metric, 0) for method in methods]
        
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=values,
                name=metric.replace('_', ' ').title(),
                marker_color=colors[i % len(colors)]
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Performance Metrics Comparison"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Indian Legal Document Retrieval System</h1>', unsafe_allow_html=True)
    st.markdown("### Compare 4 Similarity Methods for Legal Document Search")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 System Controls")
        
        # Initialize system
        if st.button("🚀 Initialize System", type="primary"):
            initialize_system()
        
        # Document upload
        st.markdown("### 📄 Upload Legal Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or Word documents",
            type=['pdf', 'docx', 'doc'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("📥 Process Uploaded Documents"):
                with st.spinner("Processing uploaded documents..."):
                    processor = DocumentProcessor()
                    new_documents = []
                    
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Process document
                            sections = processor.process_document(tmp_path)
                            new_documents.extend(sections)
                            
                            # Clean up temporary file
                            os.unlink(tmp_path)
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                    
                    if new_documents:
                        # Add to embedding manager
                        if st.session_state.embedding_manager:
                            st.session_state.embedding_manager.add_documents(new_documents)
                            st.session_state.documents.extend(new_documents)
                            st.success(f"Added {len(new_documents)} new document sections!")
                        else:
                            st.error("Please initialize the system first!")
        
        # System info
        if st.session_state.embedding_manager:
            st.markdown("### 📊 System Status")
            st.success("✅ System Initialized")
            st.info(f"📚 Documents: {len(st.session_state.documents)}")
            st.info(f"🔍 Embedding Model: {st.session_state.embedding_manager.model_name}")
        
        # Sample queries
        st.markdown("### 🔍 Sample Queries")
        sample_queries = get_sample_queries()
        selected_query = st.selectbox("Choose a sample query:", [""] + sample_queries)
        
        if selected_query:
            st.session_state.current_query = selected_query
    
    # Main content
    if st.session_state.embedding_manager is None:
        st.info("👈 Please initialize the system using the sidebar to get started!")
        return
    
    # Query input
    st.markdown("## 🔍 Search Legal Documents")
    
    # Query input with sample query option
    query = st.text_area(
        "Enter your legal query:",
        value=st.session_state.get('current_query', ''),
        height=100,
        placeholder="e.g., Income tax deduction for education expenses"
    )
    
    # Search button
    if st.button("🔍 Search Documents", type="primary") and query.strip():
        with st.spinner("Searching documents using all similarity methods..."):
            # Perform search
            results = perform_search(query.strip())
            st.session_state.search_results = results
            
            # Store query
            st.session_state.current_query = query.strip()
    
    # Display results
    if st.session_state.search_results:
        st.markdown("## 📋 Search Results")
        display_search_results(st.session_state.search_results)
        
        # Metrics dashboard
        display_metrics_dashboard(st.session_state.search_results, st.session_state.current_query)
        
        # Export results
        st.markdown("## 📤 Export Results")
        if st.button("💾 Export Results to CSV"):
            # Prepare data for export
            export_data = []
            for method_name, method_results in st.session_state.search_results.items():
                for i, (idx, score, doc) in enumerate(method_results):
                    export_data.append({
                        'Method': method_name,
                        'Rank': i + 1,
                        'Score': score,
                        'Document': doc.get('document_name', 'Unknown'),
                        'Section': doc.get('title', 'Unknown'),
                        'Category': doc.get('category', 'Unknown'),
                        'Content': doc.get('content', '')[:500] + "..." if len(doc.get('content', '')) > 500 else doc.get('content', '')
                    })
            
            if export_data:
                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"legal_search_results_{st.session_state.current_query.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main() 