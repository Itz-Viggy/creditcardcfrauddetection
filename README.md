# Credit Card Fraud Detection System

## Project Overview
A comprehensive machine learning system for detecting fraudulent credit card transactions using advanced ML algorithms and interactive visualization. The system processes 284,807 transactions with 492 fraud cases (0.17% fraud rate) and achieves 89-92% fraud detection rate at 1% false positive threshold.

## 🎯 Key Features
- **Advanced ML Pipeline**: Logistic Regression, XGBoost, and LightGBM models
- **Robust Data Preprocessing**: SMOTE resampling, feature scaling, and class imbalance handling
- **Interactive Dashboard**: Real-time fraud probability visualization and geographic heatmaps
- **Comprehensive Evaluation**: ROC AUC, Precision-Recall curves, and Recall@FPR metrics
- **Feature Engineering**: Time-based features, amount analysis, and synthetic location data

## 📊 Model Performance
| Model               | Precision | Recall | F1-Score | ROC AUC | Recall@1%FPR |
|---------------------|-----------|--------|----------|---------|--------------|
| Logistic Regression | 0.0235    | 0.9388 | 0.0458   | 0.9646  | 0.8980       |
| XGBoost             | 0.3235    | 0.8980 | 0.4757   | 0.9762  | 0.9184       |
| LightGBM            | 0.1488    | 0.9184 | 0.2560   | 0.9776  | 0.9184       |

## 🏗️ Project Structure
```
├── data/                   # Raw and processed data
├── notebooks/              # Jupyter notebooks for EDA
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── models/                 # Saved model files
├── dashboard/              # Streamlit dashboard
├── requirements.txt        # Dependencies
└── main.py                # Main execution script
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Itz-Viggy/creditcardcfrauddetection.git
   cd creditcardcfrauddetection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   ```bash
   python download_data.py
   ```

4. Run the complete pipeline:
   ```bash
   python main.py
   ```

5. Launch the interactive dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

## 📈 Technical Implementation

### Data Processing
- **Dataset**: 284,807 credit card transactions with 492 fraud cases
- **Class Imbalance**: 0.17% fraud rate handled with SMOTE resampling
- **Feature Engineering**: Time-based features, amount analysis, synthetic location data
- **Preprocessing**: Robust scaling, outlier detection, and data validation

### Machine Learning Models
- **Logistic Regression**: Baseline model with balanced class weights
- **XGBoost**: Gradient boosting with optimized hyperparameters
- **LightGBM**: Light gradient boosting for improved performance

### Evaluation Metrics
- **ROC AUC**: 0.96+ across all models
- **Recall@1%FPR**: 89-92% fraud detection at low false positive rate
- **Precision-Recall**: Optimized for imbalanced classification
- **Cross-validation**: 5-fold stratified validation

### Dashboard Features
- **Real-time Visualization**: Fraud probability distributions
- **Geographic Analysis**: Interactive heatmaps of fraud locations
- **Model Comparison**: Performance metrics and feature importance
- **Threshold Analysis**: Dynamic fraud detection sensitivity

## 🛠️ Technologies Used
- **Python**: Core programming language
- **scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost & LightGBM**: Advanced gradient boosting models
- **pandas & numpy**: Data manipulation and numerical computing
- **Streamlit**: Interactive web dashboard
- **imbalanced-learn**: Class imbalance handling
- **matplotlib & plotly**: Data visualization
- **Folium**: Geographic mapping

## 📋 Requirements
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.6.0
lightgbm>=3.3.0
streamlit>=1.15.0
plotly>=5.10.0
folium>=0.14.0
imbalanced-learn>=0.9.0
```

## 🔍 Dataset
The system uses the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains:
- **284,807 transactions** from European credit card holders
- **492 fraudulent transactions** (0.17% fraud rate)
- **28 anonymized features** (V1-V28) from PCA transformation
- **Time and Amount** features for additional analysis

