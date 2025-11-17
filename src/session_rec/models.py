"""Session-based recommendation models."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod
import logging
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for session-based recommendation models."""
    
    @abstractmethod
    def fit(self, sequences: List[List[str]], targets: List[str]) -> None:
        """Fit the model to training data.
        
        Args:
            sequences: List of session sequences.
            targets: List of target items.
        """
        pass
    
    @abstractmethod
    def predict(self, sequence: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Predict next items for a given sequence.
        
        Args:
            sequence: Input session sequence.
            top_k: Number of top recommendations.
            
        Returns:
            List of (item_id, score) tuples.
        """
        pass


class PopularityModel(BaseModel):
    """Popularity-based baseline model."""
    
    def __init__(self):
        """Initialize the popularity model."""
        self.item_counts: Dict[str, int] = {}
        self.total_interactions = 0
        
    def fit(self, sequences: List[List[str]], targets: List[str]) -> None:
        """Fit the popularity model.
        
        Args:
            sequences: List of session sequences.
            targets: List of target items.
        """
        # Count item frequencies
        self.item_counts = Counter(targets)
        self.total_interactions = sum(self.item_counts.values())
        
        logger.info(f"Fitted popularity model with {len(self.item_counts)} items")
    
    def predict(self, sequence: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Predict next items based on popularity.
        
        Args:
            sequence: Input session sequence.
            top_k: Number of top recommendations.
            
        Returns:
            List of (item_id, score) tuples.
        """
        # Get popularity scores
        scores = {
            item: count / self.total_interactions 
            for item, count in self.item_counts.items()
        }
        
        # Sort by score and return top_k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]


class ItemKNNModel(BaseModel):
    """Item-based collaborative filtering model."""
    
    def __init__(self, k: int = 20, similarity_metric: str = 'cosine'):
        """Initialize the ItemKNN model.
        
        Args:
            k: Number of nearest neighbors.
            similarity_metric: Similarity metric ('cosine' or 'jaccard').
        """
        self.k = k
        self.similarity_metric = similarity_metric
        self.item_similarities: Dict[str, Dict[str, float]] = {}
        self.item_counts: Dict[str, int] = {}
        
    def fit(self, sequences: List[List[str]], targets: List[str]) -> None:
        """Fit the ItemKNN model.
        
        Args:
            sequences: List of session sequences.
            targets: List of target items.
        """
        # Build item co-occurrence matrix
        item_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for seq, target in zip(sequences, targets):
            for item in seq:
                item_cooccurrence[item][target] += 1
                item_cooccurrence[target][item] += 1
        
        # Calculate similarities
        all_items = set()
        for seq in sequences:
            all_items.update(seq)
        all_items.update(targets)
        
        for item1 in all_items:
            self.item_similarities[item1] = {}
            for item2 in all_items:
                if item1 != item2:
                    if self.similarity_metric == 'cosine':
                        sim = self._cosine_similarity(item_cooccurrence[item1], item_cooccurrence[item2])
                    elif self.similarity_metric == 'jaccard':
                        sim = self._jaccard_similarity(item_cooccurrence[item1], item_cooccurrence[item2])
                    else:
                        raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
                    
                    if sim > 0:
                        self.item_similarities[item1][item2] = sim
        
        logger.info(f"Fitted ItemKNN model with {len(all_items)} items")
    
    def _cosine_similarity(self, vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
        """Calculate cosine similarity between two item vectors."""
        all_keys = set(vec1.keys()) | set(vec2.keys())
        if not all_keys:
            return 0.0
        
        v1 = np.array([vec1.get(k, 0) for k in all_keys])
        v2 = np.array([vec2.get(k, 0) for k in all_keys])
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _jaccard_similarity(self, vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
        """Calculate Jaccard similarity between two item vectors."""
        set1 = set(vec1.keys())
        set2 = set(vec2.keys())
        
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def predict(self, sequence: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Predict next items using item-based collaborative filtering.
        
        Args:
            sequence: Input session sequence.
            top_k: Number of top recommendations.
            
        Returns:
            List of (item_id, score) tuples.
        """
        if not sequence:
            return []
        
        # Get similarities for the last item in sequence
        last_item = sequence[-1]
        if last_item not in self.item_similarities:
            return []
        
        # Calculate scores based on similarities
        scores = {}
        for similar_item, similarity in self.item_similarities[last_item].items():
            if similar_item not in sequence:  # Don't recommend already seen items
                scores[similar_item] = similarity
        
        # Sort by score and return top_k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]


class SessionDataset(Dataset):
    """PyTorch dataset for session sequences."""
    
    def __init__(self, sequences: List[List[str]], targets: List[str], item_to_idx: Dict[str, int]):
        """Initialize the dataset.
        
        Args:
            sequences: List of session sequences.
            targets: List of target items.
            item_to_idx: Mapping from item_id to index.
        """
        self.sequences = sequences
        self.targets = targets
        self.item_to_idx = item_to_idx
        self.n_items = len(item_to_idx)
        
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item at index.
        
        Args:
            idx: Index.
            
        Returns:
            Tuple of (sequence_tensor, target_tensor).
        """
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        # Convert to indices
        seq_indices = [self.item_to_idx[item] for item in sequence]
        target_idx = self.item_to_idx[target]
        
        return torch.tensor(seq_indices, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)


class GRU4RecModel(BaseModel):
    """GRU4Rec model for session-based recommendations."""
    
    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 50,
        hidden_dim: int = 100,
        n_layers: int = 1,
        dropout: float = 0.25,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 10
    ):
        """Initialize the GRU4Rec model.
        
        Args:
            n_items: Number of unique items.
            embedding_dim: Embedding dimension.
            hidden_dim: Hidden dimension.
            n_layers: Number of GRU layers.
            dropout: Dropout rate.
            learning_rate: Learning rate.
            batch_size: Batch size.
            n_epochs: Number of training epochs.
        """
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        
    def _build_model(self) -> nn.Module:
        """Build the GRU4Rec model."""
        class GRU4Rec(nn.Module):
            def __init__(self, n_items, embedding_dim, hidden_dim, n_layers, dropout):
                super().__init__()
                self.embedding = nn.Embedding(n_items, embedding_dim)
                self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
                self.dropout = nn.Dropout(dropout)
                self.output = nn.Linear(hidden_dim, n_items)
                
            def forward(self, x):
                embedded = self.embedding(x)
                output, _ = self.gru(embedded)
                # Use the last output
                last_output = output[:, -1, :]
                dropped = self.dropout(last_output)
                return self.output(dropped)
        
        return GRU4Rec(self.n_items, self.embedding_dim, self.hidden_dim, self.n_layers, self.dropout)
    
    def fit(self, sequences: List[List[str]], targets: List[str]) -> None:
        """Fit the GRU4Rec model.
        
        Args:
            sequences: List of session sequences.
            targets: List of target items.
        """
        # Create item mappings
        all_items = set()
        for seq in sequences:
            all_items.update(seq)
        all_items.update(targets)
        
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(all_items))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Update n_items
        self.n_items = len(self.item_to_idx)
        
        # Build model
        self.model = self._build_model().to(self.device)
        
        # Create dataset and dataloader
        dataset = SessionDataset(sequences, targets, self.item_to_idx)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch_sequences, batch_targets in dataloader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("GRU4Rec model training completed")
    
    def predict(self, sequence: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Predict next items using GRU4Rec.
        
        Args:
            sequence: Input session sequence.
            top_k: Number of top recommendations.
            
        Returns:
            List of (item_id, score) tuples.
        """
        if not sequence or not self.model:
            return []
        
        # Convert to indices
        seq_indices = [self.item_to_idx[item] for item in sequence if item in self.item_to_idx]
        if not seq_indices:
            return []
        
        # Prepare input
        seq_tensor = torch.tensor([seq_indices], dtype=torch.long).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(seq_tensor)
            scores = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Get top_k items
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            item = self.idx_to_item[idx]
            if item not in sequence:  # Don't recommend already seen items
                results.append((item, float(scores[idx])))
        
        return results[:top_k]


class SASRecModel(BaseModel):
    """Self-Attentive Sequential Recommendation model."""
    
    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 50,
        n_heads: int = 1,
        n_layers: int = 2,
        dropout: float = 0.5,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 10,
        max_length: int = 50
    ):
        """Initialize the SASRec model.
        
        Args:
            n_items: Number of unique items.
            embedding_dim: Embedding dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            dropout: Dropout rate.
            learning_rate: Learning rate.
            batch_size: Batch size.
            n_epochs: Number of training epochs.
            max_length: Maximum sequence length.
        """
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_length = max_length
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        
    def _build_model(self) -> nn.Module:
        """Build the SASRec model."""
        class SASRec(nn.Module):
            def __init__(self, n_items, embedding_dim, n_heads, n_layers, dropout, max_length):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.max_length = max_length
                
                # Embeddings
                self.item_embedding = nn.Embedding(n_items, embedding_dim)
                self.pos_embedding = nn.Embedding(max_length, embedding_dim)
                
                # Transformer layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=n_heads,
                    dim_feedforward=embedding_dim * 4,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
                
                # Output layer
                self.output = nn.Linear(embedding_dim, n_items)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                seq_len = x.size(1)
                
                # Item embeddings
                item_emb = self.item_embedding(x)
                
                # Position embeddings
                pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
                pos_emb = self.pos_embedding(pos)
                
                # Combine embeddings
                emb = item_emb + pos_emb
                emb = self.dropout(emb)
                
                # Transformer
                output = self.transformer(emb)
                
                # Use the last output
                last_output = output[:, -1, :]
                return self.output(last_output)
        
        return SASRec(self.n_items, self.embedding_dim, self.n_heads, self.n_layers, self.dropout, self.max_length)
    
    def fit(self, sequences: List[List[str]], targets: List[str]) -> None:
        """Fit the SASRec model.
        
        Args:
            sequences: List of session sequences.
            targets: List of target items.
        """
        # Create item mappings
        all_items = set()
        for seq in sequences:
            all_items.update(seq)
        all_items.update(targets)
        
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(all_items))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Update n_items
        self.n_items = len(self.item_to_idx)
        
        # Build model
        self.model = self._build_model().to(self.device)
        
        # Prepare data with padding
        padded_sequences = []
        for seq in sequences:
            seq_indices = [self.item_to_idx[item] for item in seq if item in self.item_to_idx]
            if len(seq_indices) > self.max_length:
                seq_indices = seq_indices[-self.max_length:]
            else:
                seq_indices = [0] * (self.max_length - len(seq_indices)) + seq_indices
            padded_sequences.append(seq_indices)
        
        # Create dataset and dataloader
        dataset = SessionDataset(padded_sequences, targets, self.item_to_idx)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch_sequences, batch_targets in dataloader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("SASRec model training completed")
    
    def predict(self, sequence: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Predict next items using SASRec.
        
        Args:
            sequence: Input session sequence.
            top_k: Number of top recommendations.
            
        Returns:
            List of (item_id, score) tuples.
        """
        if not sequence or not self.model:
            return []
        
        # Convert to indices and pad
        seq_indices = [self.item_to_idx[item] for item in sequence if item in self.item_to_idx]
        if not seq_indices:
            return []
        
        if len(seq_indices) > self.max_length:
            seq_indices = seq_indices[-self.max_length:]
        else:
            seq_indices = [0] * (self.max_length - len(seq_indices)) + seq_indices
        
        # Prepare input
        seq_tensor = torch.tensor([seq_indices], dtype=torch.long).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(seq_tensor)
            scores = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Get top_k items
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            item = self.idx_to_item[idx]
            if item not in sequence:  # Don't recommend already seen items
                results.append((item, float(scores[idx])))
        
        return results[:top_k]


# Placeholder for BERT4Rec - would require more complex implementation
class BERT4RecModel(BaseModel):
    """BERT4Rec model placeholder - simplified implementation."""
    
    def __init__(self, **kwargs):
        """Initialize BERT4Rec model."""
        logger.warning("BERT4Rec model is a placeholder. Using SASRec instead.")
        self.sasrec = SASRecModel(**kwargs)
    
    def fit(self, sequences: List[List[str]], targets: List[str]) -> None:
        """Fit the model."""
        self.sasrec.fit(sequences, targets)
    
    def predict(self, sequence: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Predict next items."""
        return self.sasrec.predict(sequence, top_k)
