"""Unit tests for session-based recommendation system."""

import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from typing import List, Dict
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from session_rec.data import DataLoader, SessionDataGenerator
from session_rec.models import (
    PopularityModel,
    ItemKNNModel,
    GRU4RecModel,
    SASRecModel,
)
from session_rec.evaluation import Metrics, Evaluator
from session_rec.utils import set_seed, load_config, save_config


class TestSessionDataGenerator:
    """Test cases for SessionDataGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = SessionDataGenerator(n_sessions=10, n_items=5, n_users=3)
        assert generator.n_sessions == 10
        assert generator.n_items == 5
        assert generator.n_users == 3
        assert len(generator.item_popularity) == 5
        assert len(generator.user_preferences) == 3
    
    def test_generate_sessions(self):
        """Test session generation."""
        generator = SessionDataGenerator(n_sessions=5, n_items=3, n_users=2, seed=42)
        sessions_df = generator.generate_sessions()
        
        assert len(sessions_df) > 0
        assert 'session_id' in sessions_df.columns
        assert 'user_id' in sessions_df.columns
        assert 'item_id' in sessions_df.columns
        assert 'timestamp' in sessions_df.columns
        assert 'weight' in sessions_df.columns
        
        # Check session lengths
        session_lengths = sessions_df.groupby('session_id').size()
        assert all(length >= 3 for length in session_lengths)  # min session length
    
    def test_generate_items_metadata(self):
        """Test item metadata generation."""
        generator = SessionDataGenerator(n_items=5)
        items_df = generator.generate_items_metadata()
        
        assert len(items_df) == 5
        assert 'item_id' in items_df.columns
        assert 'title' in items_df.columns
        assert 'category' in items_df.columns
        assert 'price' in items_df.columns
        assert 'rating' in items_df.columns


class TestDataLoader:
    """Test cases for DataLoader."""
    
    def test_init(self):
        """Test data loader initialization."""
        loader = DataLoader("test_data")
        assert loader.data_path.name == "test_data"
    
    def test_prepare_session_data(self):
        """Test session data preparation."""
        # Create test data
        data = [
            ['session_1', 'item_1'],
            ['session_1', 'item_2'],
            ['session_1', 'item_3'],
            ['session_2', 'item_1'],
            ['session_2', 'item_4'],
        ]
        df = pd.DataFrame(data, columns=['session_id', 'item_id'])
        
        loader = DataLoader()
        sequences, targets, item_to_idx = loader.prepare_session_data(df)
        
        assert len(sequences) > 0
        assert len(targets) > 0
        assert len(item_to_idx) > 0
        assert len(sequences) == len(targets)
    
    def test_train_test_split(self):
        """Test train-test split."""
        sequences = [['item_1', 'item_2'], ['item_3', 'item_4'], ['item_5', 'item_6']]
        targets = ['item_3', 'item_5', 'item_7']
        
        loader = DataLoader()
        train_seq, train_tar, test_seq, test_tar = loader.train_test_split(
            sequences, targets, test_ratio=0.33, random_state=42
        )
        
        assert len(train_seq) + len(test_seq) == len(sequences)
        assert len(train_tar) + len(test_tar) == len(targets)
        assert len(train_seq) == len(train_tar)
        assert len(test_seq) == len(test_tar)


class TestPopularityModel:
    """Test cases for PopularityModel."""
    
    def test_init(self):
        """Test model initialization."""
        model = PopularityModel()
        assert model.item_counts == {}
        assert model.total_interactions == 0
    
    def test_fit(self):
        """Test model fitting."""
        model = PopularityModel()
        sequences = [['item_1', 'item_2'], ['item_2', 'item_3']]
        targets = ['item_3', 'item_1']
        
        model.fit(sequences, targets)
        
        assert model.total_interactions == 2
        assert 'item_3' in model.item_counts
        assert 'item_1' in model.item_counts
    
    def test_predict(self):
        """Test model prediction."""
        model = PopularityModel()
        sequences = [['item_1', 'item_2'], ['item_2', 'item_3']]
        targets = ['item_3', 'item_1', 'item_3']  # item_3 appears twice
        
        model.fit(sequences, targets)
        recommendations = model.predict(['item_1'], top_k=2)
        
        assert len(recommendations) <= 2
        assert all(isinstance(item, str) for item, _ in recommendations)
        assert all(isinstance(score, float) for _, score in recommendations)


class TestItemKNNModel:
    """Test cases for ItemKNNModel."""
    
    def test_init(self):
        """Test model initialization."""
        model = ItemKNNModel(k=10, similarity_metric='cosine')
        assert model.k == 10
        assert model.similarity_metric == 'cosine'
    
    def test_fit(self):
        """Test model fitting."""
        model = ItemKNNModel(k=5)
        sequences = [['item_1', 'item_2'], ['item_2', 'item_3'], ['item_1', 'item_3']]
        targets = ['item_3', 'item_1', 'item_2']
        
        model.fit(sequences, targets)
        
        assert len(model.item_similarities) > 0
        assert 'item_1' in model.item_similarities
    
    def test_predict(self):
        """Test model prediction."""
        model = ItemKNNModel(k=5)
        sequences = [['item_1', 'item_2'], ['item_2', 'item_3'], ['item_1', 'item_3']]
        targets = ['item_3', 'item_1', 'item_2']
        
        model.fit(sequences, targets)
        recommendations = model.predict(['item_1'], top_k=3)
        
        assert len(recommendations) <= 3
        assert all(isinstance(item, str) for item, _ in recommendations)
        assert all(isinstance(score, float) for _, score in recommendations)


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        recommended = ['item_1', 'item_2', 'item_3']
        relevant = ['item_1', 'item_3']
        
        precision = Metrics.precision_at_k(recommended, relevant, k=3)
        assert precision == 2/3  # 2 relevant out of 3 recommended
    
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        recommended = ['item_1', 'item_2', 'item_3']
        relevant = ['item_1', 'item_3', 'item_4']
        
        recall = Metrics.recall_at_k(recommended, relevant, k=3)
        assert recall == 2/3  # 2 relevant out of 3 total relevant
    
    def test_map_at_k(self):
        """Test MAP@K calculation."""
        recommended = ['item_1', 'item_2', 'item_3']
        relevant = ['item_1', 'item_3']
        
        map_score = Metrics.map_at_k(recommended, relevant, k=3)
        assert map_score > 0
        assert map_score <= 1
    
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        recommended = ['item_1', 'item_2', 'item_3']
        relevant = ['item_1', 'item_3']
        
        ndcg = Metrics.ndcg_at_k(recommended, relevant, k=3)
        assert ndcg > 0
        assert ndcg <= 1
    
    def test_hit_rate_at_k(self):
        """Test Hit Rate@K calculation."""
        recommended = ['item_1', 'item_2', 'item_3']
        relevant = ['item_2']
        
        hit_rate = Metrics.hit_rate_at_k(recommended, relevant, k=3)
        assert hit_rate == 1.0  # item_2 is in top 3
    
    def test_coverage(self):
        """Test coverage calculation."""
        recommended_all = [['item_1', 'item_2'], ['item_2', 'item_3']]
        all_items = ['item_1', 'item_2', 'item_3', 'item_4']
        
        coverage = Metrics.coverage(recommended_all, all_items)
        assert coverage == 3/4  # 3 out of 4 items covered


class TestEvaluator:
    """Test cases for Evaluator."""
    
    def test_init(self):
        """Test evaluator initialization."""
        evaluator = Evaluator(k_values=[5, 10])
        assert evaluator.k_values == [5, 10]
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create mock model
        class MockModel:
            def predict(self, sequence, top_k=10):
                return [('item_1', 0.8), ('item_2', 0.6), ('item_3', 0.4)]
        
        model = MockModel()
        evaluator = Evaluator(k_values=[5, 10])
        
        test_sequences = [['item_1'], ['item_2']]
        test_targets = ['item_1', 'item_2']
        all_items = ['item_1', 'item_2', 'item_3']
        
        results = evaluator.evaluate_model(
            model, test_sequences, test_targets, all_items
        )
        
        assert 'precision@5' in results
        assert 'recall@5' in results
        assert 'ndcg@5' in results
        assert all(0 <= score <= 1 for score in results.values() if isinstance(score, (int, float)))


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True
    
    def test_load_config(self):
        """Test config loading."""
        # Create a temporary config file
        import tempfile
        import os
        
        config_content = """
data:
  n_sessions: 100
  n_items: 50
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config['data']['n_sessions'] == 100
            assert config['data']['n_items'] == 50
        finally:
            os.unlink(temp_path)
    
    def test_save_config(self):
        """Test config saving."""
        import tempfile
        import os
        
        config = {'test': {'value': 42}}
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(config, temp_path)
            loaded_config = load_config(temp_path)
            assert loaded_config['test']['value'] == 42
        finally:
            os.unlink(temp_path)


# Integration tests
class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data generation to evaluation."""
        # Generate data
        generator = SessionDataGenerator(n_sessions=10, n_items=5, n_users=3, seed=42)
        interactions_df = generator.generate_sessions()
        
        # Prepare data
        loader = DataLoader()
        sequences, targets, item_to_idx = loader.prepare_session_data(interactions_df)
        train_seq, train_tar, test_seq, test_tar = loader.train_test_split(
            sequences, targets, test_ratio=0.3, random_state=42
        )
        
        # Train model
        model = PopularityModel()
        model.fit(train_seq, train_tar)
        
        # Evaluate
        evaluator = Evaluator(k_values=[5])
        all_items = list(item_to_idx.keys())
        results = evaluator.evaluate_model(model, test_seq, test_tar, all_items)
        
        assert 'precision@5' in results
        assert 'recall@5' in results
        assert 'ndcg@5' in results


if __name__ == "__main__":
    pytest.main([__file__])
