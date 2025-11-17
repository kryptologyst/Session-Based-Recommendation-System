#!/usr/bin/env python3
"""Main training and evaluation script for session-based recommendations."""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from session_rec.data import DataLoader, SessionDataGenerator
from session_rec.models import (
    PopularityModel,
    ItemKNNModel,
    GRU4RecModel,
    SASRecModel,
    BERT4RecModel,
)
from session_rec.evaluation import Evaluator
from session_rec.utils import set_seed, load_config, save_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for training and evaluating models."""
    parser = argparse.ArgumentParser(description='Session-based Recommendation System')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file path')
    parser.add_argument('--data_path', type=str, default='data', help='Data directory path')
    parser.add_argument('--output_path', type=str, default='results', help='Output directory path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--generate_data', action='store_true', help='Generate synthetic data')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        logger.warning(f"Config file {config_path} not found. Using default configuration.")
        config = get_default_config()
        save_config(config, str(config_path))
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader(args.data_path)
    
    # Generate or load data
    if args.generate_data:
        logger.info("Generating synthetic session data...")
        generator = SessionDataGenerator(
            n_sessions=config['data']['n_sessions'],
            n_items=config['data']['n_items'],
            n_users=config['data']['n_users'],
            session_length_range=tuple(config['data']['session_length_range']),
            seed=args.seed
        )
        
        # Generate and save data
        interactions_df = generator.generate_sessions()
        items_df = generator.generate_items_metadata()
        
        interactions_df.to_csv(data_loader.data_path / "interactions.csv", index=False)
        items_df.to_csv(data_loader.data_path / "items.csv", index=False)
        
        logger.info(f"Generated {len(interactions_df)} interactions and {len(items_df)} items")
    
    # Load data
    logger.info("Loading data...")
    interactions_df = data_loader.load_interactions()
    items_df = data_loader.load_items()
    
    # Prepare session data
    logger.info("Preparing session data...")
    sequences, targets, item_to_idx = data_loader.prepare_session_data(
        interactions_df,
        min_session_length=config['data']['min_session_length'],
        max_session_length=config['data']['max_session_length']
    )
    
    # Train-test split
    train_sequences, train_targets, test_sequences, test_targets = data_loader.train_test_split(
        sequences, targets, 
        test_ratio=config['data']['test_ratio'],
        random_state=args.seed
    )
    
    logger.info(f"Training samples: {len(train_sequences)}")
    logger.info(f"Test samples: {len(test_sequences)}")
    logger.info(f"Total items: {len(item_to_idx)}")
    
    # Initialize models
    models = {}
    
    # Popularity baseline
    models['Popularity'] = PopularityModel()
    
    # ItemKNN
    models['ItemKNN'] = ItemKNNModel(
        k=config['models']['itemknn']['k'],
        similarity_metric=config['models']['itemknn']['similarity_metric']
    )
    
    # GRU4Rec
    models['GRU4Rec'] = GRU4RecModel(
        n_items=len(item_to_idx),
        embedding_dim=config['models']['gru4rec']['embedding_dim'],
        hidden_dim=config['models']['gru4rec']['hidden_dim'],
        n_layers=config['models']['gru4rec']['n_layers'],
        dropout=config['models']['gru4rec']['dropout'],
        learning_rate=config['models']['gru4rec']['learning_rate'],
        batch_size=config['models']['gru4rec']['batch_size'],
        n_epochs=config['models']['gru4rec']['n_epochs']
    )
    
    # SASRec
    models['SASRec'] = SASRecModel(
        n_items=len(item_to_idx),
        embedding_dim=config['models']['sasrec']['embedding_dim'],
        n_heads=config['models']['sasrec']['n_heads'],
        n_layers=config['models']['sasrec']['n_layers'],
        dropout=config['models']['sasrec']['dropout'],
        learning_rate=config['models']['sasrec']['learning_rate'],
        batch_size=config['models']['sasrec']['batch_size'],
        n_epochs=config['models']['sasrec']['n_epochs'],
        max_length=config['models']['sasrec']['max_length']
    )
    
    # Train models
    logger.info("Training models...")
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        try:
            model.fit(train_sequences, train_targets)
            logger.info(f"{model_name} training completed")
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    # Evaluate models
    logger.info("Evaluating models...")
    evaluator = Evaluator(k_values=config['evaluation']['k_values'])
    
    # Calculate item popularity for novelty metric
    all_items = list(item_to_idx.keys())
    item_counts = {}
    for target in train_targets:
        item_counts[target] = item_counts.get(target, 0) + 1
    
    total_interactions = sum(item_counts.values())
    item_popularity = {
        item: count / total_interactions 
        for item, count in item_counts.items()
    }
    
    # Compare models
    results_df = evaluator.compare_models(
        models, test_sequences, test_targets, all_items, item_popularity,
        top_k=config['evaluation']['top_k']
    )
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard(
        results_df, 
        primary_metric=config['evaluation']['primary_metric']
    )
    
    # Save results
    results_df.to_csv(output_path / "evaluation_results.csv", index=False)
    leaderboard.to_csv(output_path / "leaderboard.csv", index=False)
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(leaderboard.to_string(index=False))
    
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save model configurations
    model_configs = {}
    for model_name, model in models.items():
        if hasattr(model, 'item_to_idx'):
            model_configs[model_name] = {
                'item_to_idx': model.item_to_idx,
                'n_items': len(model.item_to_idx)
            }
    
    save_config(model_configs, str(output_path / "model_configs.yaml"))
    
    logger.info("Training and evaluation completed!")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'data': {
            'n_sessions': 1000,
            'n_items': 100,
            'n_users': 200,
            'session_length_range': [3, 20],
            'min_session_length': 2,
            'max_session_length': 50,
            'test_ratio': 0.2
        },
        'models': {
            'itemknn': {
                'k': 20,
                'similarity_metric': 'cosine'
            },
            'gru4rec': {
                'embedding_dim': 50,
                'hidden_dim': 100,
                'n_layers': 1,
                'dropout': 0.25,
                'learning_rate': 0.001,
                'batch_size': 32,
                'n_epochs': 10
            },
            'sasrec': {
                'embedding_dim': 50,
                'n_heads': 1,
                'n_layers': 2,
                'dropout': 0.5,
                'learning_rate': 0.001,
                'batch_size': 32,
                'n_epochs': 10,
                'max_length': 50
            }
        },
        'evaluation': {
            'k_values': [5, 10, 20],
            'top_k': 20,
            'primary_metric': 'ndcg@10'
        }
    }


if __name__ == "__main__":
    main()
