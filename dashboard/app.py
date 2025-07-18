"""
Streamlit Dashboard for Credit Card Fraud Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import sys
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

@st.cache_data
def load_dashboard_data():
    """Load data for dashboard"""
    try:
        # Load model comparison
        model_comparison = pd.read_csv("models/model_comparison.csv")
        
        # Load feature importance
        feature_importance = pd.read_csv("models/feature_importance.csv")
        
        # Load predictions
        predictions = pd.read_csv("data/fraud_predictions.csv")
        
        # Load top fraudulent transactions
        top_fraud = pd.read_csv("data/top_fraudulent_transactions.csv")
        
        return model_comparison, feature_importance, predictions, top_fraud
    
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.error("Please run 'python main.py' first to generate the required data files.")
        return None, None, None, None

def create_fraud_overview(predictions):
    """Create fraud detection overview metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(predictions)
    actual_fraud = predictions['actual_fraud'].sum()
    predicted_fraud = predictions['predicted_fraud'].sum()
    correctly_identified = ((predictions['actual_fraud'] == 1) & 
                           (predictions['predicted_fraud'] == 1)).sum()
    
    with col1:
        st.metric(
            label="Total Transactions", 
            value=f"{total_transactions:,}"
        )
    
    with col2:
        st.metric(
            label="Actual Fraud Cases", 
            value=f"{actual_fraud:,}",
            delta=f"{(actual_fraud/total_transactions)*100:.2f}% of total"
        )
    
    with col3:
        st.metric(
            label="Predicted Fraud Cases", 
            value=f"{predicted_fraud:,}",
            delta=f"{(predicted_fraud/total_transactions)*100:.2f}% of total"
        )
    
    with col4:
        fraud_detection_rate = (correctly_identified / actual_fraud * 100) if actual_fraud > 0 else 0
        st.metric(
            label="Fraud Detection Rate", 
            value=f"{fraud_detection_rate:.1f}%",
            delta=f"{correctly_identified}/{actual_fraud} cases caught"
        )

def create_model_comparison_chart(model_comparison):
    """Create model performance comparison chart"""
    
    # Select numeric columns for comparison
    numeric_cols = ['Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Avg Precision']
    
    # Convert string values to float
    for col in numeric_cols:
        model_comparison[col] = model_comparison[col].astype(float)
    
    # Create radar chart
    fig = go.Figure()
    
    for idx, row in model_comparison.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[col] for col in numeric_cols],
            theta=numeric_cols,
            fill='toself',
            name=row['Model'],
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Comparison",
        height=500
    )
    
    return fig

def create_feature_importance_chart(feature_importance):
    """Create feature importance chart"""
    
    # Create horizontal bar chart
    fig = px.bar(
        feature_importance.head(15), 
        x='importance', 
        y='feature',
        orientation='h',
        title="Top 15 Most Important Features",
        labels={'importance': 'Feature Importance', 'feature': 'Feature Name'}
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_fraud_probability_distribution(predictions):
    """Create fraud probability distribution chart"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Fraud Probability Distribution', 'Fraud Cases by Probability Threshold'],
        vertical_spacing=0.1
    )
    
    # Histogram of fraud probabilities
    fig.add_trace(
        go.Histogram(
            x=predictions[predictions['actual_fraud'] == 0]['fraud_probability'],
            name='Normal Transactions',
            opacity=0.7,
            nbinsx=50
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=predictions[predictions['actual_fraud'] == 1]['fraud_probability'],
            name='Fraudulent Transactions',
            opacity=0.7,
            nbinsx=50
        ),
        row=1, col=1
    )
    
    # Threshold analysis
    thresholds = np.arange(0.1, 1.0, 0.1)
    fraud_caught = []
    false_positives = []
    
    for threshold in thresholds:
        predicted_fraud_at_threshold = predictions['fraud_probability'] >= threshold
        fraud_caught.append(
            ((predictions['actual_fraud'] == 1) & predicted_fraud_at_threshold).sum()
        )
        false_positives.append(
            ((predictions['actual_fraud'] == 0) & predicted_fraud_at_threshold).sum()
        )
    
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=fraud_caught,
            mode='lines+markers',
            name='Fraud Cases Caught',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=false_positives,
            mode='lines+markers',
            name='False Positives',
            line=dict(color='red'),
            yaxis='y2'
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=700, showlegend=True)
    fig.update_xaxes(title_text="Probability Threshold", row=2, col=1)
    fig.update_yaxes(title_text="Number of Cases", row=2, col=1)
    
    return fig

def create_geographic_fraud_map(predictions):
    """Create geographic visualization of fraud cases"""
    
    if 'latitude' not in predictions.columns or 'longitude' not in predictions.columns:
        st.warning("Geographic data not available")
        return None
    
    # Filter out invalid coordinates
    valid_coords = predictions.dropna(subset=['latitude', 'longitude'])
    
    if len(valid_coords) == 0:
        st.warning("No valid geographic coordinates found")
        return None
    
    # Create base map
    center_lat = valid_coords['latitude'].mean()
    center_lon = valid_coords['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    
    # Add fraud cases
    fraud_cases = valid_coords[valid_coords['actual_fraud'] == 1]
    normal_cases = valid_coords[valid_coords['actual_fraud'] == 0].sample(
        n=min(200, len(valid_coords[valid_coords['actual_fraud'] == 0])), 
        random_state=42
    )  # Sample normal cases for better visualization
    
    # Add normal transactions (blue)
    for idx, row in normal_cases.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            popup=f"Normal Transaction<br>Amount: ${row.get('Amount', 'N/A'):.2f}<br>Probability: {row['fraud_probability']:.3f}",
            color='blue',
            fillColor='lightblue',
            fillOpacity=0.6
        ).add_to(m)
    
    # Add fraud cases (red)
    for idx, row in fraud_cases.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=f"FRAUD DETECTED<br>Amount: ${row.get('Amount', 'N/A'):.2f}<br>Probability: {row['fraud_probability']:.3f}",
            color='red',
            fillColor='darkred',
            fillOpacity=0.8
        ).add_to(m)
    
    return m

def create_transaction_timeline(predictions):
    """Create timeline view of transactions"""
    
    if 'Time' not in predictions.columns:
        st.warning("Time data not available for timeline")
        return None
    
    # Convert time to hours
    predictions_with_time = predictions.copy()
    predictions_with_time['hour'] = (predictions_with_time['Time'] / 3600) % 24
    
    # Group by hour
    hourly_stats = predictions_with_time.groupby('hour').agg({
        'actual_fraud': ['count', 'sum'],
        'fraud_probability': 'mean'
    }).round(3)
    
    hourly_stats.columns = ['total_transactions', 'fraud_cases', 'avg_fraud_probability']
    hourly_stats = hourly_stats.reset_index()
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Transactions by Hour of Day', 'Average Fraud Probability by Hour'],
        shared_xaxes=True
    )
    
    # Transaction volume
    fig.add_trace(
        go.Bar(
            x=hourly_stats['hour'],
            y=hourly_stats['total_transactions'],
            name='Total Transactions',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=hourly_stats['hour'],
            y=hourly_stats['fraud_cases'],
            name='Fraud Cases',
            marker_color='red'
        ),
        row=1, col=1
    )
    
    # Average fraud probability
    fig.add_trace(
        go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['avg_fraud_probability'],
            mode='lines+markers',
            name='Avg Fraud Probability',
            line=dict(color='orange', width=3)
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
    fig.update_yaxes(title_text="Number of Transactions", row=1, col=1)
    fig.update_yaxes(title_text="Fraud Probability", row=2, col=1)
    
    return fig

def main():
    """Main dashboard function"""
    
    # Title and header
    st.title("🔒 Credit Card Fraud Detection Dashboard")
    st.markdown("### Phase 1 MVP - Batch Fraud Detection System")
    
    # Load data
    with st.spinner("Loading data..."):
        model_comparison, feature_importance, predictions, top_fraud = load_dashboard_data()
    
    if model_comparison is None:
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Overview", "🤖 Model Performance", "🔍 Feature Analysis", 
         "🗺️ Geographic View", "⏱️ Time Analysis", "🚨 Top Fraud Cases"]
    )
    
    # Main content based on selected page
    if page == "📊 Overview":
        st.header("Fraud Detection Overview")
        
        create_fraud_overview(predictions)
        
        st.subheader("Fraud Probability Distribution")
        fig = create_fraud_probability_distribution(predictions)
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detection Statistics")
            fraud_stats = predictions.groupby('actual_fraud').agg({
                'fraud_probability': ['mean', 'std', 'min', 'max']
            }).round(4)
            st.dataframe(fraud_stats)
        
        with col2:
            st.subheader("Model Accuracy Metrics")
            accuracy_data = {
                'Metric': ['True Positives', 'False Positives', 'True Negatives', 'False Negatives'],
                'Count': [
                    ((predictions['actual_fraud'] == 1) & (predictions['predicted_fraud'] == 1)).sum(),
                    ((predictions['actual_fraud'] == 0) & (predictions['predicted_fraud'] == 1)).sum(),
                    ((predictions['actual_fraud'] == 0) & (predictions['predicted_fraud'] == 0)).sum(),
                    ((predictions['actual_fraud'] == 1) & (predictions['predicted_fraud'] == 0)).sum(),
                ]
            }
            st.dataframe(pd.DataFrame(accuracy_data))
    
    elif page == "🤖 Model Performance":
        st.header("Model Performance Comparison")
        
        # Performance comparison table
        st.subheader("Model Comparison Table")
        st.dataframe(model_comparison, use_container_width=True)
        
        # Radar chart
        st.subheader("Performance Radar Chart")
        fig = create_model_comparison_chart(model_comparison)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        best_model_idx = model_comparison['Avg Precision'].astype(float).idxmax()
        best_model = model_comparison.iloc[best_model_idx]
        
        st.success(f"🏆 Best Model: **{best_model['Model']}** (Average Precision: {best_model['Avg Precision']})")
    
    elif page == "🔍 Feature Analysis":
        st.header("Feature Importance Analysis")
        
        if feature_importance is not None:
            fig = create_feature_importance_chart(feature_importance)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Feature Importance Table")
            st.dataframe(feature_importance, use_container_width=True)
        else:
            st.warning("Feature importance data not available")
    
    elif page == "🗺️ Geographic View":
        st.header("Geographic Distribution of Fraud")
        
        fraud_map = create_geographic_fraud_map(predictions)
        if fraud_map:
            st.subheader("Fraud Cases Map")
            st.markdown("🔴 Red markers: Fraud cases | 🔵 Blue markers: Normal transactions (sample)")
            st_folium(fraud_map, width=700, height=500)
            
            # Geographic statistics
            if 'city' in predictions.columns:
                st.subheader("Fraud by City")
                city_stats = predictions.groupby('city').agg({
                    'actual_fraud': ['count', 'sum', 'mean']
                }).round(3)
                city_stats.columns = ['total_transactions', 'fraud_cases', 'fraud_rate']
                city_stats = city_stats.sort_values('fraud_cases', ascending=False)
                st.dataframe(city_stats.head(10))
        else:
            st.info("Geographic visualization not available - please ensure location data is generated")
    
    elif page == "⏱️ Time Analysis":
        st.header("Time-Based Fraud Patterns")
        
        fig = create_transaction_timeline(predictions)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Time-based insights
            if 'Time' in predictions.columns:
                st.subheader("Key Time-Based Insights")
                predictions_with_hour = predictions.copy()
                predictions_with_hour['hour'] = (predictions_with_hour['Time'] / 3600) % 24
                
                peak_fraud_hour = predictions_with_hour.groupby('hour')['actual_fraud'].mean().idxmax()
                lowest_fraud_hour = predictions_with_hour.groupby('hour')['actual_fraud'].mean().idxmin()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Peak Fraud Hour", f"{int(peak_fraud_hour)}:00")
                with col2:
                    st.metric("Lowest Fraud Hour", f"{int(lowest_fraud_hour)}:00")
        else:
            st.info("Time analysis not available - please ensure time data is included")
    
    elif page == "🚨 Top Fraud Cases":
        st.header("Top Fraudulent Transactions")
        
        if top_fraud is not None:
            st.subheader("Highest Risk Transactions")
            
            # Display columns for the table
            display_columns = ['fraud_probability', 'actual_fraud', 'predicted_fraud']
            if 'Amount' in top_fraud.columns:
                display_columns.append('Amount')
            if 'city' in top_fraud.columns:
                display_columns.append('city')
            
            # Show top fraud cases
            st.dataframe(
                top_fraud[display_columns].head(20),
                use_container_width=True
            )
            
            # Fraud probability distribution for top cases
            st.subheader("Risk Score Distribution")
            fig = px.histogram(
                top_fraud.head(50),
                x='fraud_probability',
                title="Distribution of Fraud Probabilities (Top 50 Cases)",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Top fraud cases data not available")
    
    # Footer
    st.markdown("---")
    st.markdown("**Credit Card Fraud Detection System** | Phase 1 MVP | Built with Streamlit")

if __name__ == "__main__":
    main()
