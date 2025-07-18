"""
Utility functions for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple, Any
import joblib
import os

def load_data(file_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load the credit card fraud dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Run 'python download_data.py' first to download the dataset")
        return None

def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive information about the dataset"""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicate_rows': df.duplicated().sum(),
    }
    
    if 'Class' in df.columns:
        class_dist = df['Class'].value_counts()
        info['class_distribution'] = {
            'normal': int(class_dist[0]),
            'fraud': int(class_dist[1]),
            'fraud_rate': float(class_dist[1] / len(df) * 100)
        }
    
    return info

def plot_class_distribution(df: pd.DataFrame, save_path: str = None):
    """Plot the distribution of fraud vs normal transactions"""
    plt.figure(figsize=(10, 6))
    
    # Count plot
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='Class', palette=['skyblue', 'red'])
    plt.title('Transaction Class Distribution')
    plt.xlabel('Class (0: Normal, 1: Fraud)')
    plt.ylabel('Count')
    
    # Add percentage labels
    total = len(df)
    for i, v in enumerate(df['Class'].value_counts().values):
        plt.text(i, v + total*0.01, f'{v:,}\n({v/total*100:.2f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    class_counts = df['Class'].value_counts()
    plt.pie(class_counts.values, labels=['Normal', 'Fraud'], 
            autopct='%1.2f%%', colors=['skyblue', 'red'])
    plt.title('Transaction Class Proportion')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_distributions(df: pd.DataFrame, features: list = None, save_path: str = None):
    """Plot distributions of selected features by class"""
    if features is None:
        # Select a few key features for visualization
        features = ['Time', 'Amount'] + [col for col in df.columns if col.startswith('V')][:6]
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4 * n_rows))
    
    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # Plot distributions for both classes
        df[df['Class'] == 0][feature].hist(bins=50, alpha=0.7, label='Normal', color='skyblue')
        df[df['Class'] == 1][feature].hist(bins=50, alpha=0.7, label='Fraud', color='red')
        
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """Plot correlation matrix heatmap"""
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model"):
    """Comprehensive model evaluation"""
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Evaluation Results")
    print(f"{'='*50}")
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🎯 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              Normal  Fraud")
    print(f"Actual Normal  {cm[0,0]:6d}  {cm[0,1]:5d}")
    print(f"       Fraud   {cm[1,0]:6d}  {cm[1,1]:5d}")
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📏 Key Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    if y_prob is not None:
        auc_score = roc_auc_score(y_true, y_prob)
        print(f"ROC AUC: {auc_score:.4f}")
        
        # Average Precision (better for imbalanced datasets)
        avg_precision = average_precision_score(y_true, y_prob)
        print(f"Average Precision: {avg_precision:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'confusion_matrix': cm,
        'auc': auc_score if y_prob is not None else None,
        'avg_precision': avg_precision if y_prob is not None else None
    }

def plot_roc_curve(y_true, y_prob, model_name="Model", save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curve(y_true, y_prob, model_name="Model", save_path=None):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, 
             label=f'{model_name} (AP = {avg_precision:.3f})')
    
    # Baseline (proportion of positive class)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='r', linestyle='--', 
                label=f'Baseline (AP = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_model(model, model_name: str, metrics: dict = None):
    """Save trained model and its metrics"""
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = f"models/{model_name}_model.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save metrics
    if metrics:
        metrics_path = f"models/{model_name}_metrics.joblib"
        joblib.dump(metrics, metrics_path)
        print(f"✅ Metrics saved: {metrics_path}")

def load_model(model_name: str):
    """Load saved model and metrics"""
    model_path = f"models/{model_name}_model.joblib"
    metrics_path = f"models/{model_name}_metrics.joblib"
    
    model = None
    metrics = None
    
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded: {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found: {model_path}")
    
    try:
        metrics = joblib.load(metrics_path)
        print(f"✅ Metrics loaded: {metrics_path}")
    except FileNotFoundError:
        print(f"⚠️  Metrics not found: {metrics_path}")
    
    return model, metrics

def create_synthetic_locations(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic location data for geographic visualization
    This simulates realistic transaction locations
    """
    np.random.seed(seed)
    
    df_with_location = df.copy()
    
    # Define major city coordinates (lat, lon)
    cities = {
        'New York': (40.7128, -74.0060),
        'Los Angeles': (34.0522, -118.2437),
        'Chicago': (41.8781, -87.6298),
        'Houston': (29.7604, -95.3698),
        'Phoenix': (33.4484, -112.0740),
        'Philadelphia': (39.9526, -75.1652),
        'San Antonio': (29.4241, -98.4936),
        'San Diego': (32.7157, -117.1611),
        'Dallas': (32.7767, -96.7970),
        'San Jose': (37.3382, -121.8863),
        'Austin': (30.2672, -97.7431),
        'Jacksonville': (30.3322, -81.6557),
        'Fort Worth': (32.7555, -97.3308),
        'Columbus': (39.9612, -82.9988),
        'Charlotte': (35.2271, -80.8431),
        'San Francisco': (37.7749, -122.4194),
        'Indianapolis': (39.7684, -86.1581),
        'Seattle': (47.6062, -122.3321),
        'Denver': (39.7392, -104.9903),
        'Washington DC': (38.9072, -77.0369)
    }
    
    city_names = list(cities.keys())
    city_coords = list(cities.values())
    
    # Assign cities with different probabilities (urban areas more likely)
    city_weights = [0.15, 0.12, 0.08, 0.06, 0.04, 0.05, 0.04, 0.04, 0.05, 0.04,
                   0.03, 0.03, 0.03, 0.03, 0.03, 0.06, 0.03, 0.04, 0.04, 0.05]
    
    # Normalize weights to ensure they sum to 1.0
    city_weights = np.array(city_weights)
    city_weights = city_weights / city_weights.sum()
    
    # Sample cities for each transaction
    chosen_cities = np.random.choice(len(city_names), size=len(df), p=city_weights)
    
    # Add some random noise to coordinates (within ~10km)
    noise_scale = 0.1  # roughly 10km in degrees
    
    latitudes = []
    longitudes = []
    city_labels = []
    
    for city_idx in chosen_cities:
        base_lat, base_lon = city_coords[city_idx]
        
        # Add random noise
        lat = base_lat + np.random.normal(0, noise_scale)
        lon = base_lon + np.random.normal(0, noise_scale)
        
        latitudes.append(lat)
        longitudes.append(lon)
        city_labels.append(city_names[city_idx])
    
    df_with_location['latitude'] = latitudes
    df_with_location['longitude'] = longitudes
    df_with_location['city'] = city_labels
    
    # Fraudulent transactions might have slightly different geographic patterns
    # (This is just for simulation - in real data, you'd analyze actual patterns)
    fraud_mask = df_with_location['Class'] == 1
    if fraud_mask.sum() > 0:
        # Add slight bias towards certain locations for fraud (simulation)
        fraud_indices = df_with_location[fraud_mask].index
        for idx in fraud_indices:
            if np.random.random() < 0.3:  # 30% chance to modify location
                # Bias towards airports, tourist areas (simulate card skimming)
                df_with_location.loc[idx, 'latitude'] += np.random.normal(0, 0.05)
                df_with_location.loc[idx, 'longitude'] += np.random.normal(0, 0.05)
    
    print(f"✅ Added synthetic location data to {len(df_with_location)} transactions")
    print(f"📍 Cities represented: {len(set(city_labels))}")
    
    return df_with_location
