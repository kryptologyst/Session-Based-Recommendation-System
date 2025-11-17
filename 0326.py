#!/usr/bin/env python3
"""
Project 326: Session-based Recommendations - Modern Implementation

This is a modernized version of the original session-based recommendation system.
The original simple implementation has been replaced with a comprehensive system
that includes multiple models, proper evaluation, and production-ready code.

For the full implementation, see:
- src/session_rec/ - Core package with models and evaluation
- scripts/train.py - Training and evaluation script
- scripts/demo.py - Interactive Streamlit demo
- notebooks/session_rec_analysis.ipynb - Analysis notebook

Quick start:
    python scripts/train.py --generate_data
    streamlit run scripts/demo.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from session_rec.data import DataLoader, SessionDataGenerator
from session_rec.models import PopularityModel, ItemKNNModel
from session_rec.evaluation import Evaluator
from session_rec.utils import set_seed


def simple_demo():
    """Simple demo showing the basic functionality."""
    print("Session-Based Recommendation System - Simple Demo")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Generate small dataset
    generator = SessionDataGenerator(n_sessions=10, n_items=5, n_users=3, seed=42)
    interactions_df = generator.generate_sessions()
    
    print(f"Generated {len(interactions_df)} interactions")
    print("\nSample data:")
    print(interactions_df.head())
    
    # Prepare data
    data_loader = DataLoader()
    sequences, targets, item_to_idx = data_loader.prepare_session_data(interactions_df)
    
    # Train simple models
    popularity_model = PopularityModel()
    popularity_model.fit(sequences, targets)
    
    itemknn_model = ItemKNNModel(k=5)
    itemknn_model.fit(sequences, targets)
    
    # Get recommendations
    test_session = sequences[0]
    print(f"\nTest session: {test_session}")
    
    pop_recs = popularity_model.predict(test_session, top_k=3)
    knn_recs = itemknn_model.predict(test_session, top_k=3)
    
    print(f"\nPopularity recommendations: {[item for item, _ in pop_recs]}")
    print(f"ItemKNN recommendations: {[item for item, _ in knn_recs]}")
    
    print("\n" + "=" * 50)
    print("For full functionality, run:")
    print("  python scripts/train.py --generate_data")
    print("  streamlit run scripts/demo.py")


if __name__ == "__main__":
    simple_demo()