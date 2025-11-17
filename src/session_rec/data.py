"""Data loading and generation utilities for session-based recommendations."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import logging
from collections import defaultdict

from .utils import set_seed

logger = logging.getLogger(__name__)


class SessionDataGenerator:
    """Generate realistic session-based interaction data."""
    
    def __init__(
        self,
        n_sessions: int = 1000,
        n_items: int = 100,
        n_users: int = 200,
        session_length_range: Tuple[int, int] = (3, 20),
        seed: int = 42
    ):
        """Initialize the data generator.
        
        Args:
            n_sessions: Number of sessions to generate.
            n_items: Number of unique items.
            n_users: Number of unique users.
            session_length_range: Range of session lengths (min, max).
            seed: Random seed for reproducibility.
        """
        self.n_sessions = n_sessions
        self.n_items = n_items
        self.n_users = n_users
        self.session_length_range = session_length_range
        self.seed = seed
        
        set_seed(seed)
        
        # Generate item popularity distribution (power law)
        self.item_popularity = np.random.power(0.5, n_items)
        self.item_popularity = self.item_popularity / self.item_popularity.sum()
        
        # Generate user preferences
        self.user_preferences = np.random.dirichlet(np.ones(n_items), n_users)
        
    def generate_sessions(self) -> pd.DataFrame:
        """Generate session-based interaction data.
        
        Returns:
            DataFrame with columns: session_id, user_id, item_id, timestamp, weight
        """
        sessions_data = []
        
        for session_id in range(self.n_sessions):
            # Random session length
            session_length = np.random.randint(
                self.session_length_range[0], 
                self.session_length_range[1] + 1
            )
            
            # Random user for this session
            user_id = np.random.randint(0, self.n_users)
            
            # Generate items for this session with some coherence
            session_items = []
            
            # First item based on user preference
            first_item = np.random.choice(
                self.n_items, 
                p=self.user_preferences[user_id]
            )
            session_items.append(first_item)
            
            # Subsequent items with some coherence (similar items more likely)
            for _ in range(session_length - 1):
                # Mix of user preference and item popularity
                item_probs = (
                    0.3 * self.user_preferences[user_id] + 
                    0.7 * self.item_popularity
                )
                
                # Avoid duplicates in session
                available_items = [i for i in range(self.n_items) if i not in session_items]
                if not available_items:
                    break
                    
                item_probs = item_probs[available_items]
                item_probs = item_probs / item_probs.sum()
                
                next_item = np.random.choice(available_items, p=item_probs)
                session_items.append(next_item)
            
            # Add timestamp progression
            for i, item_id in enumerate(session_items):
                sessions_data.append({
                    'session_id': f'session_{session_id}',
                    'user_id': f'user_{user_id}',
                    'item_id': f'item_{item_id}',
                    'timestamp': session_id * 1000 + i,  # Simple timestamp
                    'weight': 1.0
                })
        
        return pd.DataFrame(sessions_data)
    
    def generate_items_metadata(self) -> pd.DataFrame:
        """Generate item metadata.
        
        Returns:
            DataFrame with item metadata.
        """
        items_data = []
        
        # Generate categories
        categories = ['electronics', 'clothing', 'books', 'home', 'sports', 'beauty']
        
        for item_id in range(self.n_items):
            items_data.append({
                'item_id': f'item_{item_id}',
                'title': f'Item {item_id}',
                'category': np.random.choice(categories),
                'price': np.random.uniform(10, 500),
                'rating': np.random.uniform(3.0, 5.0),
                'description': f'Description for item {item_id}'
            })
        
        return pd.DataFrame(items_data)


class DataLoader:
    """Data loader for session-based recommendation data."""
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize the data loader.
        
        Args:
            data_path: Path to the data directory.
        """
        self.data_path = Path(data_path) if data_path else Path("data")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def load_interactions(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load interaction data.
        
        Args:
            file_path: Path to interactions file.
            
        Returns:
            DataFrame with interaction data.
        """
        if file_path is None:
            file_path = self.data_path / "interactions.csv"
        
        if not Path(file_path).exists():
            logger.warning(f"File {file_path} not found. Generating synthetic data.")
            generator = SessionDataGenerator()
            df = generator.generate_sessions()
            df.to_csv(file_path, index=False)
            return df
        
        return pd.read_csv(file_path)
    
    def load_items(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load item metadata.
        
        Args:
            file_path: Path to items file.
            
        Returns:
            DataFrame with item metadata.
        """
        if file_path is None:
            file_path = self.data_path / "items.csv"
        
        if not Path(file_path).exists():
            logger.warning(f"File {file_path} not found. Generating synthetic data.")
            generator = SessionDataGenerator()
            df = generator.generate_items_metadata()
            df.to_csv(file_path, index=False)
            return df
        
        return pd.read_csv(file_path)
    
    def prepare_session_data(
        self, 
        interactions_df: pd.DataFrame,
        min_session_length: int = 2,
        max_session_length: int = 50
    ) -> Tuple[List[List[str]], List[str], Dict[str, int]]:
        """Prepare session data for training.
        
        Args:
            interactions_df: DataFrame with interaction data.
            min_session_length: Minimum session length to include.
            max_session_length: Maximum session length to include.
            
        Returns:
            Tuple of (session_sequences, session_targets, item_to_idx)
        """
        # Group by session
        sessions = interactions_df.groupby('session_id')['item_id'].apply(list).to_dict()
        
        # Filter sessions by length
        filtered_sessions = {
            sid: items for sid, items in sessions.items()
            if min_session_length <= len(items) <= max_session_length
        }
        
        # Create sequences and targets
        sequences = []
        targets = []
        
        for session_items in filtered_sessions.values():
            # Create sequences of different lengths
            for i in range(1, len(session_items)):
                sequences.append(session_items[:i])
                targets.append(session_items[i])
        
        # Create item mapping
        all_items = set()
        for items in sessions.values():
            all_items.update(items)
        
        item_to_idx = {item: idx for idx, item in enumerate(sorted(all_items))}
        
        return sequences, targets, item_to_idx
    
    def train_test_split(
        self,
        sequences: List[List[str]],
        targets: List[str],
        test_ratio: float = 0.2,
        random_state: int = 42
    ) -> Tuple[List[List[str]], List[str], List[List[str]], List[str]]:
        """Split data into train and test sets.
        
        Args:
            sequences: List of session sequences.
            targets: List of target items.
            test_ratio: Ratio of test data.
            random_state: Random seed.
            
        Returns:
            Tuple of (train_sequences, train_targets, test_sequences, test_targets)
        """
        np.random.seed(random_state)
        n_samples = len(sequences)
        test_size = int(n_samples * test_ratio)
        
        # Random shuffle
        indices = np.random.permutation(n_samples)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        train_sequences = [sequences[i] for i in train_indices]
        train_targets = [targets[i] for i in train_indices]
        test_sequences = [sequences[i] for i in test_indices]
        test_targets = [targets[i] for i in test_indices]
        
        return train_sequences, train_targets, test_sequences, test_targets
