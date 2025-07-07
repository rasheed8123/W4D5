import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import tempfile
from datetime import datetime
import json

# Import our pipeline
from pipeline.orchestrator import IntelligentChunkingPipeline

# Page configuration
st.set_page_config(
    page_title="Intelligent Document Chunking Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .chunk-preview {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .error-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []

def initialize_pipeline():
    """Initialize the intelligent chunking pipeline."""
    if st.session_state.pipeline is None:
        with st.spinner("Initializing Intelligent Document Chunking Pipeline..."):
            try:
                # Get configuration from sidebar
                embedding_model = st.session_state.get('embedding_model', 'all-MiniLM-L6-v2')
                use_openai = st.session_state.get('use_openai', False)
                vector_store_path = st.session_state.get('vector_store_path', './chroma_db')
                
                st.session_state.pipeline = IntelligentChunkingPipeline(
                    vector_store_path=vector_store_path,
                    embedding_model=embedding_model,
                    use_openai=use_openai
                )
                
                st.success("✅ Pipeline initialized successfully!")
                
            except Exception as e:
                st.error(f"❌ Error initializing pipeline: {e}")

def process_uploaded_files(uploaded_files):
    """Process uploaded files through the pipeline."""
    if not st.session_state.pipeline:
        st.error("Please initialize the pipeline first!")
        return
    
    if not uploaded_files:
        st.warning("No files uploaded!")
        return
    
    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        
        # Save uploaded files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # Process files
        with st.spinner(f"Processing {len(file_paths)} documents..."):
            results = st.session_state.pipeline.process_multiple_documents(file_paths)
            st.session_state.processing_results.extend(results)
        
        # Display results
        display_processing_results(results)

def display_processing_results(results):
    """Display processing results."""
    if not results:
        return
    
    st.markdown("## 📊 Processing Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    successful = len([r for r in results if not r.error])
    failed = len([r for r in results if r.error])
    total_chunks = sum(r.chunk_count for r in results if not r.error)
    avg_time = sum(r.processing_time for r in results) / len(results) if results else 0
    
    with col1:
        st.metric("✅ Successful", successful, delta=None)
    with col2:
        st.metric("❌ Failed", failed, delta=None)
    with col3:
        st.metric("📄 Total Chunks", total_chunks, delta=None)
    with col4:
        st.metric("⏱️ Avg Time", f"{avg_time:.2f}s", delta=None)
    
    # Detailed results
    for i, result in enumerate(results):
        with st.expander(f"📄 {result.filename} ({result.doc_type})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Document Type:** {result.doc_type}")
                st.write(f"**Classification Confidence:** {result.classification_confidence:.3f}")
                st.write(f"**Processing Time:** {result.processing_time:.2f}s")
                st.write(f"**Chunk Count:** {result.chunk_count}")
                
                if result.error:
                    st.error(f"**Error:** {result.error}")
            
            with col2:
                if result.chunks:
                    st.write("**Chunk Preview:**")
                    for j, (chunk_content, chunk_metadata) in enumerate(result.chunks[:3]):
                        with st.expander(f"Chunk {j+1} ({chunk_metadata.chunk_type})"):
                            st.text(chunk_content[:300] + "..." if len(chunk_content) > 300 else chunk_content)
                            st.write(f"**Metadata:** {chunk_metadata.chunk_type}, {chunk_metadata.word_count} words")

def display_metrics_dashboard():
    """Display comprehensive metrics dashboard."""
    if not st.session_state.pipeline:
        return
    
    stats = st.session_state.pipeline.get_processing_stats()
    
    st.markdown("## 📈 Performance Metrics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        success_rate = stats.get('success_rate', 0)
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    with col2:
        total_docs = stats.get('total_documents', 0)
        st.metric("Total Documents", total_docs)
    
    with col3:
        total_chunks = stats.get('total_chunks', 0)
        st.metric("Total Chunks", total_chunks)
    
    with col4:
        avg_time = stats.get('avg_processing_time', 0)
        st.metric("Avg Processing Time", f"{avg_time:.2f}s")
    
    # Document type distribution
    if stats.get('doc_type_distribution'):
        st.markdown("### 📊 Document Type Distribution")
        doc_types = list(stats['doc_type_distribution'].keys())
        doc_counts = list(stats['doc_type_distribution'].values())
        
        fig = px.pie(
            values=doc_counts,
            names=doc_types,
            title="Document Types Processed"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Chunk type distribution
    if stats.get('chunk_type_distribution'):
        st.markdown("### ✂️ Chunk Type Distribution")
        chunk_types = list(stats['chunk_type_distribution'].keys())
        chunk_counts = list(stats['chunk_type_distribution'].values())
        
        fig = px.bar(
            x=chunk_types,
            y=chunk_counts,
            title="Chunk Types Generated"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Processing time distribution
    if stats.get('processing_times'):
        st.markdown("### ⏱️ Processing Time Distribution")
        fig = px.histogram(
            x=stats['processing_times'],
            title="Document Processing Times",
            labels={'x': 'Time (seconds)', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Vector store stats
    if stats.get('vector_store_stats'):
        vs_stats = stats['vector_store_stats']
        st.markdown("### 💾 Vector Store Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", vs_stats.get('total_chunks', 0))
        with col2:
            st.metric("Unique Documents", len(vs_stats.get('documents', [])))
        with col3:
            st.metric("Collections", len(vs_stats.get('available_collections', [])))

def search_interface():
    """Search interface for testing retrieval."""
    st.markdown("## 🔍 Search & Retrieval Testing")
    
    if not st.session_state.pipeline:
        st.warning("Please initialize the pipeline first!")
        return
    
    # Search query
    query = st.text_input("Enter your search query:", placeholder="e.g., API authentication methods")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_results = st.slider("Number of results:", 1, 20, 5)
    with col2:
        filter_doc_type = st.selectbox("Filter by document type:", 
                                     ["All"] + list(st.session_state.pipeline.processing_stats.get('doc_type_distribution', {}).keys()))
    with col3:
        filter_chunk_type = st.selectbox("Filter by chunk type:", 
                                       ["All"] + list(st.session_state.pipeline.processing_stats.get('chunk_type_distribution', {}).keys()))
    
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching..."):
            # Prepare filters
            filter_metadata = {}
            if filter_doc_type != "All":
                filter_metadata['doc_type'] = filter_doc_type
            if filter_chunk_type != "All":
                filter_metadata['chunk_type'] = filter_chunk_type
            
            # Perform search
            results = st.session_state.pipeline.search_documents(
                query=query,
                n_results=n_results,
                filter_metadata=filter_metadata if filter_metadata else None
            )
            
            st.session_state.search_results = results
            
            # Display results
            display_search_results(results, query)

def display_search_results(results, query):
    """Display search results."""
    if not results:
        st.warning("No results found!")
        return
    
    st.markdown(f"### 📋 Search Results for: '{query}'")
    
    # Results summary
    st.write(f"Found {len(results)} results")
    
    # Display each result
    for i, result in enumerate(results):
        with st.expander(f"Result {i+1} (Score: {1 - result.get('distance', 0):.3f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Content:**")
                st.text(result['chunk'][:500] + "..." if len(result['chunk']) > 500 else result['chunk'])
            
            with col2:
                metadata = result.get('metadata', {})
                st.write("**Metadata:**")
                st.write(f"**Document:** {metadata.get('source_document', 'Unknown')}")
                st.write(f"**Chunk Type:** {metadata.get('chunk_type', 'Unknown')}")
                st.write(f"**Word Count:** {metadata.get('word_count', 'Unknown')}")
                if metadata.get('section_header'):
                    st.write(f"**Section:** {metadata['section_header']}")

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Intelligent Document Chunking Agent</h1>', unsafe_allow_html=True)
    st.markdown("### Adaptive Document Processing for Enterprise Knowledge Bases")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Pipeline initialization
        if st.button("🚀 Initialize Pipeline", type="primary"):
            initialize_pipeline()
        
        # Configuration options
        st.markdown("### 🔧 Settings")
        
        embedding_model = st.selectbox(
            "Embedding Model:",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
            index=0
        )
        st.session_state.embedding_model = embedding_model
        
        use_openai = st.checkbox("Use OpenAI Embeddings", value=False)
        st.session_state.use_openai = use_openai
        
        vector_store_path = st.text_input("Vector Store Path:", value="./chroma_db")
        st.session_state.vector_store_path = vector_store_path
        
        # Pipeline info
        if st.session_state.pipeline:
            st.markdown("### 📊 Pipeline Status")
            st.success("✅ Pipeline Active")
            
            pipeline_info = st.session_state.pipeline.get_pipeline_info()
            st.write(f"**Embedding Model:** {pipeline_info['embedder']['model_name']}")
            st.write(f"**OpenAI:** {'✅' if pipeline_info['embedder']['use_openai'] else '❌'}")
            st.write(f"**Vector Store:** {pipeline_info['vector_store']['collection_name']}")
        
        # Actions
        st.markdown("### 🛠️ Actions")
        
        if st.button("📊 View Metrics"):
            st.session_state.show_metrics = True
        
        if st.button("🔍 Search Interface"):
            st.session_state.show_search = True
        
        if st.button("🗑️ Reset Pipeline"):
            if st.session_state.pipeline:
                st.session_state.pipeline.reset_pipeline()
                st.session_state.processing_results = []
                st.session_state.search_results = []
                st.success("Pipeline reset!")
    
    # Main content
    if st.session_state.pipeline is None:
        st.info("👈 Please initialize the pipeline using the sidebar to get started!")
        return
    
    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Metrics Dashboard", "🔍 Search & Test", "📈 Performance"])
    
    with tab1:
        st.markdown("## 📤 Document Upload & Processing")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose documents to process",
            type=['pdf', 'docx', 'doc', 'md', 'txt', 'html'],
            accept_multiple_files=True,
            help="Upload multiple documents for batch processing"
        )
        
        if uploaded_files:
            st.write(f"📁 Uploaded {len(uploaded_files)} files:")
            for file in uploaded_files:
                st.write(f"  - {file.name} ({file.size} bytes)")
            
            if st.button("🔄 Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)
        
        # Processing results
        if st.session_state.processing_results:
            st.markdown("## 📋 Recent Processing Results")
            display_processing_results(st.session_state.processing_results[-5:])  # Show last 5
    
    with tab2:
        display_metrics_dashboard()
    
    with tab3:
        search_interface()
    
    with tab4:
        st.markdown("## 📈 Advanced Performance Analysis")
        
        if st.session_state.pipeline:
            # Pipeline configuration info
            pipeline_info = st.session_state.pipeline.get_pipeline_info()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔧 Pipeline Configuration")
                st.json(pipeline_info)
            
            with col2:
                st.markdown("### 📊 Current Statistics")
                stats = st.session_state.pipeline.get_processing_stats()
                st.json(stats)
            
            # Export functionality
            st.markdown("### 📤 Export Results")
            if st.button("💾 Export Processing Results"):
                if st.session_state.processing_results:
                    export_path = f"processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    st.session_state.pipeline.export_results(export_path)
                    
                    with open(export_path, 'r') as f:
                        st.download_button(
                            label="📥 Download Results JSON",
                            data=f.read(),
                            file_name=export_path,
                            mime="application/json"
                        )

if __name__ == "__main__":
    main() 