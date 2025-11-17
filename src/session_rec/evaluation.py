"""Evaluation metrics and evaluation system for session-based recommendations."""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)


class Metrics:
    """Collection of evaluation metrics for recommendation systems."""
    
    @staticmethod
    def precision_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Precision@K.
        
        Args:
            recommended_items: List of recommended items.
            relevant_items: List of relevant items.
            k: Number of top recommendations to consider.
            
        Returns:
            Precision@K score.
        """
        if k == 0:
            return 0.0
        
        top_k_recommended = recommended_items[:k]
        relevant_in_top_k = len(set(top_k_recommended) & set(relevant_items))
        
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Recall@K.
        
        Args:
            recommended_items: List of recommended items.
            relevant_items: List of relevant items.
            k: Number of top recommendations to consider.
            
        Returns:
            Recall@K score.
        """
        if not relevant_items:
            return 0.0
        
        top_k_recommended = recommended_items[:k]
        relevant_in_top_k = len(set(top_k_recommended) & set(relevant_items))
        
        return relevant_in_top_k / len(relevant_items)
    
    @staticmethod
    def map_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Mean Average Precision@K.
        
        Args:
            recommended_items: List of recommended items.
            relevant_items: List of relevant items.
            k: Number of top recommendations to consider.
            
        Returns:
            MAP@K score.
        """
        if not relevant_items:
            return 0.0
        
        top_k_recommended = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recommended):
            if item in relevant_set:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_items) if relevant_items else 0.0
    
    @staticmethod
    def ndcg_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            recommended_items: List of recommended items.
            relevant_items: List of relevant items.
            k: Number of top recommendations to consider.
            
        Returns:
            NDCG@K score.
        """
        if k == 0:
            return 0.0
        
        top_k_recommended = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recommended):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def hit_rate_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommended_items: List of recommended items.
            relevant_items: List of relevant items.
            k: Number of top recommendations to consider.
            
        Returns:
            Hit Rate@K score (0 or 1).
        """
        top_k_recommended = recommended_items[:k]
        return 1.0 if any(item in relevant_items for item in top_k_recommended) else 0.0
    
    @staticmethod
    def coverage(recommended_items_all: List[List[str]], all_items: List[str]) -> float:
        """Calculate catalog coverage.
        
        Args:
            recommended_items_all: List of recommendation lists for all users/sessions.
            all_items: List of all available items.
            
        Returns:
            Coverage score.
        """
        if not all_items:
            return 0.0
        
        recommended_items_set = set()
        for rec_list in recommended_items_all:
            recommended_items_set.update(rec_list)
        
        return len(recommended_items_set) / len(all_items)
    
    @staticmethod
    def diversity(recommended_items: List[str], item_similarities: Optional[Dict[Tuple[str, str], float]] = None) -> float:
        """Calculate intra-list diversity.
        
        Args:
            recommended_items: List of recommended items.
            item_similarities: Optional similarity matrix between items.
            
        Returns:
            Diversity score.
        """
        if len(recommended_items) <= 1:
            return 0.0
        
        if item_similarities is None:
            # Simple diversity based on unique items
            return len(set(recommended_items)) / len(recommended_items)
        
        # Calculate average pairwise dissimilarity
        total_dissimilarity = 0.0
        count = 0
        
        for i, item1 in enumerate(recommended_items):
            for j, item2 in enumerate(recommended_items):
                if i != j:
                    similarity = item_similarities.get((item1, item2), 0.0)
                    total_dissimilarity += (1.0 - similarity)
                    count += 1
        
        return total_dissimilarity / count if count > 0 else 0.0
    
    @staticmethod
    def novelty(recommended_items: List[str], item_popularity: Dict[str, float]) -> float:
        """Calculate novelty of recommendations.
        
        Args:
            recommended_items: List of recommended items.
            item_popularity: Dictionary mapping items to their popularity scores.
            
        Returns:
            Average novelty score.
        """
        if not recommended_items:
            return 0.0
        
        novelty_scores = []
        for item in recommended_items:
            popularity = item_popularity.get(item, 0.0)
            novelty = -np.log2(popularity + 1e-8)  # Avoid log(0)
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)


class Evaluator:
    """Evaluation system for session-based recommendation models."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """Initialize the evaluator.
        
        Args:
            k_values: List of k values for evaluation metrics.
        """
        self.k_values = k_values
        self.metrics = Metrics()
        
    def evaluate_model(
        self,
        model,
        test_sequences: List[List[str]],
        test_targets: List[str],
        all_items: List[str],
        item_popularity: Optional[Dict[str, float]] = None,
        top_k: int = 20
    ) -> Dict[str, float]:
        """Evaluate a recommendation model.
        
        Args:
            model: Trained recommendation model.
            test_sequences: List of test session sequences.
            test_targets: List of test target items.
            all_items: List of all available items.
            item_popularity: Optional item popularity scores.
            top_k: Number of recommendations to generate.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        results = {}
        
        # Generate recommendations for all test sequences
        all_recommendations = []
        hit_counts = defaultdict(int)
        
        for sequence, target in zip(test_sequences, test_targets):
            try:
                recommendations = model.predict(sequence, top_k=top_k)
                recommended_items = [item for item, _ in recommendations]
                all_recommendations.append(recommended_items)
                
                # Calculate metrics for each k value
                for k in self.k_values:
                    precision = self.metrics.precision_at_k(recommended_items, [target], k)
                    recall = self.metrics.recall_at_k(recommended_items, [target], k)
                    map_score = self.metrics.map_at_k(recommended_items, [target], k)
                    ndcg = self.metrics.ndcg_at_k(recommended_items, [target], k)
                    hit_rate = self.metrics.hit_rate_at_k(recommended_items, [target], k)
                    
                    results[f'precision@{k}'] = results.get(f'precision@{k}', 0) + precision
                    results[f'recall@{k}'] = results.get(f'recall@{k}', 0) + recall
                    results[f'map@{k}'] = results.get(f'map@{k}', 0) + map_score
                    results[f'ndcg@{k}'] = results.get(f'ndcg@{k}', 0) + ndcg
                    results[f'hit_rate@{k}'] = results.get(f'hit_rate@{k}', 0) + hit_rate
                    
                    hit_counts[k] += hit_rate
                
            except Exception as e:
                logger.warning(f"Error evaluating sequence: {e}")
                continue
        
        # Average the metrics
        n_samples = len(test_sequences)
        for k in self.k_values:
            for metric in ['precision', 'recall', 'map', 'ndcg', 'hit_rate']:
                key = f'{metric}@{k}'
                if key in results:
                    results[key] /= n_samples
        
        # Calculate additional metrics
        if all_recommendations:
            results['coverage'] = self.metrics.coverage(all_recommendations, all_items)
            
            # Calculate diversity (simplified - based on unique items)
            diversity_scores = []
            for rec_list in all_recommendations:
                diversity = self.metrics.diversity(rec_list)
                diversity_scores.append(diversity)
            results['diversity'] = np.mean(diversity_scores)
            
            # Calculate novelty if popularity is provided
            if item_popularity:
                novelty_scores = []
                for rec_list in all_recommendations:
                    novelty = self.metrics.novelty(rec_list, item_popularity)
                    novelty_scores.append(novelty)
                results['novelty'] = np.mean(novelty_scores)
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, Any],
        test_sequences: List[List[str]],
        test_targets: List[str],
        all_items: List[str],
        item_popularity: Optional[Dict[str, float]] = None,
        top_k: int = 20
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            models: Dictionary mapping model names to model instances.
            test_sequences: List of test session sequences.
            test_targets: List of test target items.
            all_items: List of all available items.
            item_popularity: Optional item popularity scores.
            top_k: Number of recommendations to generate.
            
        Returns:
            DataFrame with comparison results.
        """
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            model_results = self.evaluate_model(
                model, test_sequences, test_targets, all_items, item_popularity, top_k
            )
            model_results['model'] = model_name
            results.append(model_results)
        
        return pd.DataFrame(results)
    
    def create_leaderboard(
        self,
        results_df: pd.DataFrame,
        primary_metric: str = 'ndcg@10'
    ) -> pd.DataFrame:
        """Create a model leaderboard.
        
        Args:
            results_df: DataFrame with evaluation results.
            primary_metric: Primary metric for ranking.
            
        Returns:
            Sorted DataFrame with model rankings.
        """
        if primary_metric not in results_df.columns:
            logger.warning(f"Primary metric {primary_metric} not found. Using first available metric.")
            primary_metric = results_df.columns[1]  # Skip 'model' column
        
        leaderboard = results_df.sort_values(primary_metric, ascending=False)
        leaderboard['rank'] = range(1, len(leaderboard) + 1)
        
        return leaderboard
