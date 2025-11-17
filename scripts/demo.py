"""Streamlit demo for session-based recommendation system."""

import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import yaml
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
from session_rec.evaluation import Evaluator
from session_rec.utils import set_seed

# Configure page
st.set_page_config(
    page_title="Session-Based Recommendations",
    page_icon="🛍️",
    layout="wide"
)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'data' not in st.session_state:
    st.session_state.data = {}


def load_or_generate_data():
    """Load or generate session data."""
    with st.spinner("Loading data..."):
        data_loader = DataLoader("data")
        
        # Generate synthetic data if not exists
        interactions_df = data_loader.load_interactions()
        items_df = data_loader.load_items()
        
        # Prepare session data
        sequences, targets, item_to_idx = data_loader.prepare_session_data(
            interactions_df,
            min_session_length=2,
            max_session_length=50
        )
        
        # Train-test split
        train_sequences, train_targets, test_sequences, test_targets = data_loader.train_test_split(
            sequences, targets, test_ratio=0.2, random_state=42
        )
        
        st.session_state.data = {
            'interactions_df': interactions_df,
            'items_df': items_df,
            'train_sequences': train_sequences,
            'train_targets': train_targets,
            'test_sequences': test_sequences,
            'test_targets': test_targets,
            'item_to_idx': item_to_idx,
            'all_items': list(item_to_idx.keys())
        }
        
        return True


def train_models():
    """Train recommendation models."""
    if not st.session_state.data:
        return False
    
    with st.spinner("Training models..."):
        data = st.session_state.data
        
        # Initialize models
        models = {}
        
        # Popularity baseline
        models['Popularity'] = PopularityModel()
        
        # ItemKNN
        models['ItemKNN'] = ItemKNNModel(k=20, similarity_metric='cosine')
        
        # GRU4Rec (simplified for demo)
        models['GRU4Rec'] = GRU4RecModel(
            n_items=len(data['item_to_idx']),
            embedding_dim=32,
            hidden_dim=64,
            n_layers=1,
            dropout=0.25,
            learning_rate=0.001,
            batch_size=16,
            n_epochs=3  # Reduced for demo
        )
        
        # SASRec (simplified for demo)
        models['SASRec'] = SASRecModel(
            n_items=len(data['item_to_idx']),
            embedding_dim=32,
            n_heads=1,
            n_layers=1,
            dropout=0.5,
            learning_rate=0.001,
            batch_size=16,
            n_epochs=3,  # Reduced for demo
            max_length=20
        )
        
        # Train models
        for model_name, model in models.items():
            try:
                model.fit(data['train_sequences'], data['train_targets'])
                st.success(f"✅ {model_name} trained successfully")
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        st.session_state.models = models
        st.session_state.models_trained = True
        
        return True


def main():
    """Main Streamlit app."""
    st.title("🛍️ Session-Based Recommendation System")
    st.markdown("""
    This demo showcases different session-based recommendation models including:
    - **Popularity**: Baseline model recommending most popular items
    - **ItemKNN**: Item-based collaborative filtering
    - **GRU4Rec**: RNN-based sequential model
    - **SASRec**: Self-attentive sequential model
    """)
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Data loading section
    st.sidebar.subheader("Data")
    if st.sidebar.button("Load/Generate Data"):
        if load_or_generate_data():
            st.sidebar.success("Data loaded successfully!")
        else:
            st.sidebar.error("Failed to load data")
    
    # Model training section
    st.sidebar.subheader("Models")
    if st.sidebar.button("Train Models"):
        if st.session_state.data:
            if train_models():
                st.sidebar.success("Models trained successfully!")
            else:
                st.sidebar.error("Failed to train models")
        else:
            st.sidebar.error("Please load data first")
    
    # Main content
    if not st.session_state.data:
        st.info("👆 Please load data first using the sidebar controls.")
        return
    
    data = st.session_state.data
    
    # Data overview
    st.header("📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", len(data['interactions_df']['session_id'].unique()))
    
    with col2:
        st.metric("Total Items", len(data['all_items']))
    
    with col3:
        st.metric("Training Samples", len(data['train_sequences']))
    
    with col4:
        st.metric("Test Samples", len(data['test_sequences']))
    
    # Data visualization
    st.subheader("Session Length Distribution")
    session_lengths = [len(seq) for seq in data['train_sequences']]
    fig = px.histogram(
        x=session_lengths,
        nbins=20,
        title="Distribution of Session Lengths",
        labels={'x': 'Session Length', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model evaluation
    if st.session_state.models_trained and st.session_state.models:
        st.header("🎯 Model Evaluation")
        
        # Evaluate models
        evaluator = Evaluator(k_values=[5, 10, 20])
        
        # Calculate item popularity
        item_counts = {}
        for target in data['train_targets']:
            item_counts[target] = item_counts.get(target, 0) + 1
        
        total_interactions = sum(item_counts.values())
        item_popularity = {
            item: count / total_interactions 
            for item, count in item_counts.items()
        }
        
        # Compare models
        results_df = evaluator.compare_models(
            st.session_state.models, 
            data['test_sequences'], 
            data['test_targets'], 
            data['all_items'], 
            item_popularity,
            top_k=20
        )
        
        # Display results
        st.subheader("Evaluation Results")
        
        # Create leaderboard
        leaderboard = evaluator.create_leaderboard(results_df, primary_metric='ndcg@10')
        
        # Display metrics
        metrics_to_show = ['precision@5', 'precision@10', 'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']
        available_metrics = [m for m in metrics_to_show if m in results_df.columns]
        
        if available_metrics:
            st.subheader("Performance Metrics")
            
            # Create comparison chart
            fig = go.Figure()
            
            for metric in available_metrics:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df['model'],
                    y=results_df[metric],
                    text=[f"{v:.3f}" for v in results_df[metric]],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed results table
        st.subheader("Detailed Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Recommendation demo
        st.header("🔍 Recommendation Demo")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Session Input")
            
            # Item selection
            selected_items = st.multiselect(
                "Select items for session:",
                options=data['all_items'][:50],  # Limit for demo
                default=data['all_items'][:3] if len(data['all_items']) >= 3 else []
            )
            
            # Model selection
            selected_model = st.selectbox(
                "Select model:",
                options=list(st.session_state.models.keys())
            )
            
            top_k = st.slider("Number of recommendations:", 1, 20, 10)
            
            if st.button("Get Recommendations"):
                if selected_items and selected_model in st.session_state.models:
                    model = st.session_state.models[selected_model]
                    recommendations = model.predict(selected_items, top_k=top_k)
                    
                    with col2:
                        st.subheader("Recommendations")
                        
                        if recommendations:
                            rec_df = pd.DataFrame(recommendations, columns=['Item', 'Score'])
                            rec_df['Rank'] = range(1, len(rec_df) + 1)
                            rec_df = rec_df[['Rank', 'Item', 'Score']]
                            
                            st.dataframe(rec_df, use_container_width=True)
                            
                            # Show item details
                            st.subheader("Item Details")
                            for i, (item, score) in enumerate(recommendations[:5]):
                                item_info = data['items_df'][data['items_df']['item_id'] == item]
                                if not item_info.empty:
                                    item_info = item_info.iloc[0]
                                    st.write(f"**{item}** (Score: {score:.3f})")
                                    st.write(f"Category: {item_info['category']}")
                                    st.write(f"Price: ${item_info['price']:.2f}")
                                    st.write(f"Rating: {item_info['rating']:.1f}/5.0")
                                    st.write("---")
                        else:
                            st.warning("No recommendations available")
                else:
                    st.warning("Please select items and ensure model is trained")
    
    else:
        st.info("👆 Please train models first using the sidebar controls.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Session-Based Recommendation System Demo**
    
    This demo showcases various approaches to session-based recommendations:
    - Traditional collaborative filtering methods
    - Deep learning approaches using RNNs and Transformers
    - Comprehensive evaluation metrics
    
    Built with Streamlit and PyTorch.
    """)


if __name__ == "__main__":
    main()
