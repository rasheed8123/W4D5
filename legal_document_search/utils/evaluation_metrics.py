import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter

class EvaluationMetrics:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_precision_at_k(self, relevant_docs: Set[int], retrieved_docs: List[int], k: int = 5) -> float:
        """
        Calculate Precision@k.
        
        Args:
            relevant_docs: Set of relevant document indices
            retrieved_docs: List of retrieved document indices
            k: Number of top results to consider
            
        Returns:
            Precision@k score
        """
        if k == 0:
            return 0.0
        
        top_k_docs = set(retrieved_docs[:k])
        relevant_retrieved = len(top_k_docs.intersection(relevant_docs))
        return relevant_retrieved / k
    
    def calculate_recall_at_k(self, relevant_docs: Set[int], retrieved_docs: List[int], k: int = 5) -> float:
        """
        Calculate Recall@k.
        
        Args:
            relevant_docs: Set of relevant document indices
            retrieved_docs: List of retrieved document indices
            k: Number of top results to consider
            
        Returns:
            Recall@k score
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        top_k_docs = set(retrieved_docs[:k])
        relevant_retrieved = len(top_k_docs.intersection(relevant_docs))
        return relevant_retrieved / len(relevant_docs)
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_diversity_score(self, retrieved_docs: List[Dict], method: str = "section") -> float:
        """
        Calculate diversity score based on document sections or categories.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            method: Method to calculate diversity ('section', 'document', 'category')
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if not retrieved_docs:
            return 0.0
        
        if method == "section":
            # Count unique sections
            sections = [doc.get('section_id', 'unknown') for doc in retrieved_docs]
        elif method == "document":
            # Count unique documents
            sections = [doc.get('document_name', 'unknown') for doc in retrieved_docs]
        elif method == "category":
            # Count unique categories (if available)
            sections = [doc.get('category', 'unknown') for doc in retrieved_docs]
        else:
            sections = [doc.get('section_id', 'unknown') for doc in retrieved_docs]
        
        unique_sections = len(set(sections))
        total_sections = len(sections)
        
        return unique_sections / total_sections if total_sections > 0 else 0.0
    
    def calculate_novelty_score(self, retrieved_docs: List[Dict], all_docs: List[Dict]) -> float:
        """
        Calculate novelty score based on how many new sections are introduced.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            all_docs: List of all available documents
            
        Returns:
            Novelty score (0-1, higher is more novel)
        """
        if not retrieved_docs:
            return 0.0
        
        # Get all section IDs from all documents
        all_sections = set()
        for doc in all_docs:
            all_sections.add(doc.get('section_id', ''))
        
        # Get section IDs from retrieved documents
        retrieved_sections = set()
        for doc in retrieved_docs:
            retrieved_sections.add(doc.get('section_id', ''))
        
        # Calculate novelty as proportion of new sections
        if len(all_sections) == 0:
            return 0.0
        
        return len(retrieved_sections) / len(all_sections)
    
    def calculate_coverage_score(self, retrieved_docs: List[Dict], relevant_docs: Set[int], 
                               all_docs: List[Dict]) -> float:
        """
        Calculate coverage score - how well the results cover different aspects.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            relevant_docs: Set of relevant document indices
            all_docs: List of all available documents
            
        Returns:
            Coverage score (0-1, higher is better coverage)
        """
        if not retrieved_docs or len(relevant_docs) == 0:
            return 0.0
        
        # Get unique documents in retrieved results
        retrieved_doc_names = set(doc.get('document_name', '') for doc in retrieved_docs)
        
        # Get unique documents in relevant set
        relevant_doc_names = set()
        for idx in relevant_docs:
            if idx < len(all_docs):
                relevant_doc_names.add(all_docs[idx].get('document_name', ''))
        
        # Calculate coverage as intersection over union
        if len(relevant_doc_names) == 0:
            return 0.0
        
        intersection = len(retrieved_doc_names.intersection(relevant_doc_names))
        union = len(retrieved_doc_names.union(relevant_doc_names))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_serendipity_score(self, retrieved_docs: List[Dict], query: str) -> float:
        """
        Calculate serendipity score - how surprising/interesting the results are.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            query: Search query
            
        Returns:
            Serendipity score (0-1, higher is more serendipitous)
        """
        if not retrieved_docs:
            return 0.0
        
        # Simple implementation: count documents that don't contain query words
        query_words = set(query.lower().split())
        surprising_docs = 0
        
        for doc in retrieved_docs:
            content_words = set(doc.get('content', '').lower().split())
            # If less than 50% of query words appear in content, consider it surprising
            if len(query_words.intersection(content_words)) / len(query_words) < 0.5:
                surprising_docs += 1
        
        return surprising_docs / len(retrieved_docs)
    
    def evaluate_search_results(self, query: str, results: Dict[str, List[Tuple[int, float, Dict]]], 
                              relevant_docs: Set[int] = None, all_docs: List[Dict] = None) -> Dict[str, Dict]:
        """
        Comprehensive evaluation of search results for all methods.
        
        Args:
            query: Search query
            results: Dictionary with method names as keys and results as values
            relevant_docs: Set of relevant document indices (optional)
            all_docs: List of all available documents (optional)
            
        Returns:
            Dictionary with evaluation metrics for each method
        """
        evaluation_results = {}
        
        for method_name, method_results in results.items():
            # Extract document indices and documents
            doc_indices = [idx for idx, _, _ in method_results]
            documents = [doc for _, _, doc in method_results]
            
            metrics = {}
            
            # Basic metrics
            if relevant_docs is not None:
                metrics['precision_at_5'] = self.calculate_precision_at_k(relevant_docs, doc_indices, 5)
                metrics['recall_at_5'] = self.calculate_recall_at_k(relevant_docs, doc_indices, 5)
                metrics['f1_score'] = self.calculate_f1_score(metrics['precision_at_5'], metrics['recall_at_5'])
            
            # Diversity metrics
            metrics['diversity_section'] = self.calculate_diversity_score(documents, 'section')
            metrics['diversity_document'] = self.calculate_diversity_score(documents, 'document')
            
            # Additional metrics
            if all_docs is not None:
                metrics['novelty'] = self.calculate_novelty_score(documents, all_docs)
                if relevant_docs is not None:
                    metrics['coverage'] = self.calculate_coverage_score(documents, relevant_docs, all_docs)
            
            metrics['serendipity'] = self.calculate_serendipity_score(documents, query)
            
            # Average similarity score
            if method_results:
                avg_similarity = np.mean([score for _, score, _ in method_results])
                metrics['avg_similarity'] = avg_similarity
            
            evaluation_results[method_name] = metrics
        
        # Store in history
        self.metrics_history.append({
            'query': query,
            'results': evaluation_results,
            'timestamp': np.datetime64('now')
        })
        
        return evaluation_results
    
    def get_aggregate_metrics(self) -> Dict[str, Dict]:
        """
        Calculate aggregate metrics across all evaluations.
        
        Returns:
            Dictionary with aggregate metrics for each method
        """
        if not self.metrics_history:
            return {}
        
        aggregate_metrics = {}
        method_names = set()
        
        # Collect all method names
        for evaluation in self.metrics_history:
            method_names.update(evaluation['results'].keys())
        
        # Calculate aggregates for each method
        for method_name in method_names:
            method_metrics = []
            
            for evaluation in self.metrics_history:
                if method_name in evaluation['results']:
                    method_metrics.append(evaluation['results'][method_name])
            
            if method_metrics:
                aggregate = {}
                for metric_name in method_metrics[0].keys():
                    values = [m.get(metric_name, 0) for m in method_metrics if metric_name in m]
                    if values:
                        aggregate[f'avg_{metric_name}'] = np.mean(values)
                        aggregate[f'std_{metric_name}'] = np.std(values)
                        aggregate[f'min_{metric_name}'] = np.min(values)
                        aggregate[f'max_{metric_name}'] = np.max(values)
                
                aggregate_metrics[method_name] = aggregate
        
        return aggregate_metrics
    
    def clear_history(self):
        """Clear evaluation history."""
        self.metrics_history = [] 