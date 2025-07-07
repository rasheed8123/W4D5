"""
LangChain Integration for Sales Conversion Prediction
Orchestrates the complete pipeline using LangChain components
"""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import logging
import json
import numpy as np
from pipeline.embedder import SalesEmbedder
from pipeline.scorer import ConversionPredictor, HybridScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesConversionOutputParser(BaseOutputParser):
    """Parse sales conversion prediction output"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the LLM output into structured format"""
        try:
            # Try to extract JSON from the response
            if '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                json_str = text[start:end]
                return json.loads(json_str)
            else:
                # Fallback to simple parsing
                return {
                    'conversion_likelihood': 'medium',
                    'confidence': 0.5,
                    'reasoning': text,
                    'key_factors': []
                }
        except Exception as e:
            logger.warning(f"Failed to parse output: {e}")
            return {
                'conversion_likelihood': 'unknown',
                'confidence': 0.0,
                'reasoning': text,
                'key_factors': []
            }

class SalesConversionChain:
    """LangChain-based sales conversion prediction pipeline"""
    
    def __init__(self, model_path: str = None, 
                 generic_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_path = model_path
        self.generic_model_name = generic_model_name
        self.embedder = None
        self.predictor = None
        self.vector_store = None
        self.chain = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing sales conversion pipeline components...")
        
        # Initialize embedder
        self.embedder = SalesEmbedder(
            model_path=self.model_path,
            model_name=self.generic_model_name
        )
        
        logger.info("Pipeline components initialized")
    
    def load_training_data(self, csv_path: str):
        """Load and prepare training data"""
        import pandas as pd
        
        logger.info(f"Loading training data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        transcripts = df['call_transcript'].tolist()
        labels = df['conversion_label'].values
        
        # Compute embeddings
        embeddings = self.embedder.encode(transcripts)
        
        # Initialize predictor
        self.predictor = ConversionPredictor(
            embeddings=embeddings,
            labels=labels,
            transcripts=transcripts,
            scorer_type='hybrid'
        )
        
        # Create vector store for similarity search
        self._create_vector_store(transcripts, embeddings)
        
        logger.info(f"Loaded {len(transcripts)} training examples")
    
    def _create_vector_store(self, transcripts: List[str], embeddings: np.ndarray):
        """Create FAISS vector store for similarity search"""
        # Convert embeddings to documents
        documents = []
        for i, transcript in enumerate(transcripts):
            doc = Document(
                page_content=transcript,
                metadata={
                    'transcript_id': i,
                    'embedding': embeddings[i].tolist()
                }
            )
            documents.append(doc)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(
            documents,
            self.embedder.model
        )
        
        logger.info("Vector store created successfully")
    
    def create_analysis_chain(self) -> LLMChain:
        """Create LangChain for detailed analysis"""
        
        # Define the prompt template
        analysis_prompt = PromptTemplate(
            input_variables=["transcript", "conversion_score", "similar_cases", "confidence"],
            template="""
            Analyze the following sales call transcript and provide insights on conversion likelihood.
            
            TRANSCRIPT:
            {transcript}
            
            CONVERSION ANALYSIS:
            - Conversion Score: {conversion_score:.3f}
            - Confidence: {confidence:.3f}
            
            SIMILAR HISTORICAL CASES:
            {similar_cases}
            
            Please provide a detailed analysis including:
            1. Conversion likelihood assessment
            2. Key factors influencing the prediction
            3. Similar patterns from historical data
            4. Recommendations for improving conversion chances
            
            Format your response as JSON with the following structure:
            {{
                "conversion_likelihood": "high/medium/low",
                "confidence": 0.0-1.0,
                "key_factors": ["factor1", "factor2", ...],
                "similar_patterns": ["pattern1", "pattern2", ...],
                "recommendations": ["rec1", "rec2", ...],
                "reasoning": "detailed explanation"
            }}
            
            ANALYSIS:
            """
        )
        
        # Create the chain
        self.chain = LLMChain(
            llm=None,  # Will be set when LLM is available
            prompt=analysis_prompt,
            output_parser=SalesConversionOutputParser()
        )
        
        return self.chain
    
    def predict_conversion(self, transcript: str, 
                          include_analysis: bool = True) -> Dict[str, Any]:
        """Predict conversion likelihood for a transcript"""
        
        if self.predictor is None:
            raise ValueError("Training data not loaded. Call load_training_data() first.")
        
        # Encode transcript
        embedding = self.embedder.encode(transcript)
        
        # Get prediction
        prediction = self.predictor.predict_conversion(embedding)
        
        # Find similar cases
        similar_cases = self._find_similar_cases(transcript, top_k=3)
        
        # Prepare result
        result = {
            'transcript': transcript,
            'conversion_score': prediction.conversion_score,
            'conversion_probability': prediction.conversion_probability,
            'prediction': prediction.prediction,
            'confidence': prediction.confidence,
            'similar_cases': similar_cases,
            'reasoning': prediction.reasoning
        }
        
        # Add detailed analysis if requested
        if include_analysis and self.chain:
            analysis = self._generate_analysis(transcript, prediction, similar_cases)
            result['detailed_analysis'] = analysis
        
        return result
    
    def _find_similar_cases(self, transcript: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar cases using vector store"""
        if self.vector_store is None:
            return []
        
        # Search for similar documents
        docs = self.vector_store.similarity_search(transcript, k=top_k)
        
        similar_cases = []
        for i, doc in enumerate(docs):
            similar_cases.append({
                'transcript': doc.page_content[:200] + "...",  # Truncate for display
                'similarity_score': doc.metadata.get('similarity', 0.0),
                'rank': i + 1
            })
        
        return similar_cases
    
    def _generate_analysis(self, transcript: str, prediction: Any, 
                          similar_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed analysis using LangChain"""
        if self.chain is None:
            return {}
        
        # Format similar cases for the prompt
        cases_text = ""
        for case in similar_cases:
            cases_text += f"- {case['transcript']} (Similarity: {case['similarity_score']:.3f})\n"
        
        # Prepare input for the chain
        chain_input = {
            "transcript": transcript,
            "conversion_score": prediction.conversion_score,
            "similar_cases": cases_text,
            "confidence": prediction.confidence
        }
        
        try:
            # Run the chain (if LLM is available)
            if hasattr(self.chain, 'llm') and self.chain.llm is not None:
                analysis = self.chain.run(chain_input)
                return analysis
            else:
                # Fallback analysis without LLM
                return self._fallback_analysis(prediction, similar_cases)
        except Exception as e:
            logger.warning(f"Failed to generate analysis: {e}")
            return self._fallback_analysis(prediction, similar_cases)
    
    def _fallback_analysis(self, prediction: Any, 
                          similar_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback analysis without LLM"""
        # Determine likelihood based on probability
        if prediction.conversion_probability > 0.7:
            likelihood = "high"
        elif prediction.conversion_probability > 0.4:
            likelihood = "medium"
        else:
            likelihood = "low"
        
        # Extract key factors from similar cases
        key_factors = []
        if similar_cases:
            key_factors = [
                "Similar patterns found in historical data",
                f"Confidence level: {prediction.confidence:.2f}",
                f"Conversion score: {prediction.conversion_score:.3f}"
            ]
        
        return {
            "conversion_likelihood": likelihood,
            "confidence": prediction.confidence,
            "key_factors": key_factors,
            "similar_patterns": [f"Found {len(similar_cases)} similar cases"],
            "recommendations": [
                "Review similar successful cases for patterns",
                "Focus on addressing customer pain points",
                "Follow up with personalized approach"
            ],
            "reasoning": prediction.reasoning
        }
    
    def batch_predict(self, transcripts: List[str]) -> List[Dict[str, Any]]:
        """Predict conversion for multiple transcripts"""
        results = []
        for transcript in transcripts:
            result = self.predict_conversion(transcript, include_analysis=False)
            results.append(result)
        
        return results
    
    def evaluate_pipeline(self, test_transcripts: List[str], 
                         test_labels: List[int]) -> Dict[str, float]:
        """Evaluate the complete pipeline performance"""
        if self.predictor is None:
            raise ValueError("Training data not loaded")
        
        # Encode test transcripts
        test_embeddings = self.embedder.encode(test_transcripts)
        
        # Evaluate using predictor
        metrics = self.predictor.evaluate_predictions(test_embeddings, test_labels)
        
        return metrics
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline"""
        return {
            'model_path': self.model_path,
            'generic_model_name': self.generic_model_name,
            'is_finetuned': self.embedder.is_finetuned if self.embedder else False,
            'predictor_initialized': self.predictor is not None,
            'vector_store_initialized': self.vector_store is not None,
            'chain_initialized': self.chain is not None
        }

class SalesConversionRAGChain:
    """RAG-based sales conversion analysis using LangChain"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.embedder = None
        self.vector_store = None
        self.rag_chain = None
        
        self._initialize_rag_components()
    
    def _initialize_rag_components(self):
        """Initialize RAG components"""
        logger.info("Initializing RAG components...")
        
        # Initialize embedder
        self.embedder = SalesEmbedder(model_path=self.model_path)
        
        logger.info("RAG components initialized")
    
    def load_knowledge_base(self, transcripts: List[str], 
                           metadata: List[Dict[str, Any]] = None):
        """Load transcripts into knowledge base"""
        logger.info(f"Loading {len(transcripts)} transcripts into knowledge base")
        
        # Create documents with metadata
        documents = []
        for i, transcript in enumerate(transcripts):
            doc_metadata = metadata[i] if metadata else {}
            doc_metadata['transcript_id'] = i
            
            doc = Document(
                page_content=transcript,
                metadata=doc_metadata
            )
            documents.append(doc)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(
            documents,
            self.embedder.model
        )
        
        logger.info("Knowledge base created successfully")
    
    def create_rag_chain(self, llm=None):
        """Create RAG chain for sales analysis"""
        
        # Define the prompt template
        rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a sales conversion expert. Use the following context from sales call transcripts to answer the question.
            
            CONTEXT:
            {context}
            
            QUESTION:
            {question}
            
            Please provide a detailed analysis based on the context provided. Focus on:
            1. Patterns in successful vs unsuccessful calls
            2. Key factors that influence conversion
            3. Specific recommendations based on the data
            
            ANSWER:
            """
        )
        
        # Create the RAG chain
        if llm:
            self.rag_chain = (
                {"context": self.vector_store.as_retriever(), "question": RunnablePassthrough()}
                | rag_prompt
                | llm
                | StrOutputParser()
            )
        
        return self.rag_chain
    
    def query_knowledge_base(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the knowledge base for insights"""
        if self.vector_store is None:
            raise ValueError("Knowledge base not loaded")
        
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(question, k=top_k)
        
        # Extract insights
        insights = {
            'question': question,
            'relevant_transcripts': [],
            'key_insights': [],
            'recommendations': []
        }
        
        for doc in docs:
            insights['relevant_transcripts'].append({
                'transcript': doc.page_content[:300] + "...",
                'metadata': doc.metadata
            })
        
        # Generate insights based on retrieved documents
        insights['key_insights'] = self._extract_insights(docs)
        insights['recommendations'] = self._generate_recommendations(docs)
        
        return insights
    
    def _extract_insights(self, docs: List[Document]) -> List[str]:
        """Extract key insights from retrieved documents"""
        insights = []
        
        # Analyze conversion patterns
        conversion_count = sum(1 for doc in docs if doc.metadata.get('conversion_label') == 1)
        total_count = len(docs)
        
        if total_count > 0:
            conversion_rate = conversion_count / total_count
            insights.append(f"Conversion rate in similar cases: {conversion_rate:.2%}")
        
        # Extract common themes
        common_phrases = self._find_common_phrases([doc.page_content for doc in docs])
        if common_phrases:
            insights.append(f"Common themes: {', '.join(common_phrases[:3])}")
        
        return insights
    
    def _find_common_phrases(self, texts: List[str]) -> List[str]:
        """Find common phrases in texts"""
        # Simple implementation - can be enhanced with NLP
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Return most common words (excluding stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        common_words = [word for word, count in word_counts.most_common(10) 
                       if word not in stop_words and len(word) > 3]
        
        return common_words
    
    def _generate_recommendations(self, docs: List[Document]) -> List[str]:
        """Generate recommendations based on retrieved documents"""
        recommendations = []
        
        # Analyze successful vs unsuccessful patterns
        successful_docs = [doc for doc in docs if doc.metadata.get('conversion_label') == 1]
        unsuccessful_docs = [doc for doc in docs if doc.metadata.get('conversion_label') == 0]
        
        if successful_docs:
            recommendations.append("Study successful call patterns for replication")
        
        if unsuccessful_docs:
            recommendations.append("Identify and avoid patterns from unsuccessful calls")
        
        recommendations.extend([
            "Focus on customer pain points and solutions",
            "Build rapport and trust through conversation",
            "Provide clear value propositions"
        ])
        
        return recommendations

def main():
    """Example usage of the LangChain integration"""
    # Initialize the pipeline
    pipeline = SalesConversionChain(model_path='models/finetuned_sales_model')
    
    # Load training data
    pipeline.load_training_data('data/sample_transcripts.csv')
    
    # Test prediction
    test_transcript = "Agent: Hi, I'm calling about your recent inquiry. Customer: Yes, we're interested in your solution. Agent: Great! Let me tell you about our features..."
    
    result = pipeline.predict_conversion(test_transcript)
    print("Prediction Result:", result)
    
    # Get pipeline info
    info = pipeline.get_pipeline_info()
    print("Pipeline Info:", info)

if __name__ == "__main__":
    main() 