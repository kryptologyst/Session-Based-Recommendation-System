# Session-Based Recommendation System

A comprehensive session-based recommendation system implementing multiple state-of-the-art models with proper evaluation metrics and interactive demos.

## Overview

Session-based recommendation systems focus on recommending items based on the current session or recent behavior rather than historical data from the user's past sessions. These systems are essential for platforms where users don't have much past data, such as e-commerce or video streaming.

This project implements and compares multiple approaches:
- **Baseline Models**: Popularity-based and Item-based Collaborative Filtering
- **Deep Learning Models**: GRU4Rec (RNN-based) and SASRec (Transformer-based)
- **Comprehensive Evaluation**: Multiple metrics including Precision@K, Recall@K, MAP@K, NDCG@K, Hit Rate, Coverage, Diversity, and Novelty

## Features

- **Multiple Models**: Popularity, ItemKNN, GRU4Rec, SASRec
- **Realistic Data Generation**: Synthetic session data with user preferences and item popularity
- **Comprehensive Evaluation**: Standard recommendation metrics plus diversity and novelty
- **Interactive Demo**: Streamlit-based web interface for model comparison and recommendations
- **Production Ready**: Clean code structure, type hints, documentation, and testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Session-Based-Recommendation-System.git
cd Session-Based-Recommendation-System

# Install dependencies
pip install -r requirements.txt
# or
pip install -e .
```

### Basic Usage

1. **Generate synthetic data and train models:**
```bash
python scripts/train.py --generate_data --config configs/default.yaml
```

2. **Run the interactive demo:**
```bash
streamlit run scripts/demo.py
```

3. **Evaluate specific models:**
```python
from session_rec import DataLoader, PopularityModel, GRU4RecModel, Evaluator

# Load data
data_loader = DataLoader("data")
interactions_df = data_loader.load_interactions()
sequences, targets, item_to_idx = data_loader.prepare_session_data(interactions_df)

# Train model
model = GRU4RecModel(n_items=len(item_to_idx))
model.fit(sequences, targets)

# Get recommendations
recommendations = model.predict(["item_1", "item_2"], top_k=10)
print(recommendations)
```

## Project Structure

```
session-based-recommendations/
├── src/session_rec/          # Core package
│   ├── __init__.py
│   ├── data.py              # Data loading and generation
│   ├── models.py            # Recommendation models
│   ├── evaluation.py        # Evaluation metrics
│   └── utils.py             # Utility functions
├── scripts/                 # Executable scripts
│   ├── train.py             # Training and evaluation
│   └── demo.py              # Streamlit demo
├── configs/                 # Configuration files
│   └── default.yaml
├── data/                    # Data directory
├── results/                 # Output directory
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Data Format

The system expects data in the following format:

### Interactions (`interactions.csv`)
```csv
session_id,user_id,item_id,timestamp,weight
session_1,user_1,item_1,1000,1.0
session_1,user_1,item_2,1001,1.0
...
```

### Items (`items.csv`)
```csv
item_id,title,category,price,rating,description
item_1,Item 1,electronics,99.99,4.5,Description for item 1
item_2,Item 2,clothing,29.99,4.2,Description for item 2
...
```

## Models

### Popularity Model
Simple baseline recommending most popular items globally.

### ItemKNN Model
Item-based collaborative filtering using cosine or Jaccard similarity.

### GRU4Rec Model
RNN-based sequential model using Gated Recurrent Units for session modeling.

### SASRec Model
Self-Attentive Sequential Recommendation model using Transformer architecture.

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **MAP@K**: Mean Average Precision considering ranking
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Binary metric indicating if any relevant item is recommended
- **Coverage**: Fraction of catalog items that are recommended
- **Diversity**: Intra-list diversity of recommendations
- **Novelty**: Average novelty of recommended items

## Configuration

Models and evaluation can be configured via YAML files:

```yaml
data:
  n_sessions: 1000
  n_items: 100
  session_length_range: [3, 20]

models:
  gru4rec:
    embedding_dim: 50
    hidden_dim: 100
    n_epochs: 10

evaluation:
  k_values: [5, 10, 20]
  primary_metric: ndcg@10
```

## Interactive Demo

The Streamlit demo provides:
- Data visualization and statistics
- Model training and evaluation
- Interactive recommendation interface
- Performance comparison charts
- Item details and explanations

## Development

### Code Quality
- Type hints throughout
- Google-style docstrings
- PEP8 compliance (enforced with black/ruff)
- Unit tests with pytest

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ scripts/
ruff check src/ scripts/
```

## Performance

The system is designed for scalability:
- Efficient data structures for large item catalogs
- Batch processing for deep learning models
- Configurable model complexity
- Memory-efficient session handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Hidasi, B., et al. "Session-based recommendations with recurrent neural networks." ICLR 2016.
- Kang, W. C., & McAuley, J. "Self-attentive sequential recommendation." ICDM 2018.
- Quadrana, M., et al. "Personalizing session-based recommendations with hierarchical recurrent neural networks." RecSys 2017.
# Session-Based-Recommendation-System
