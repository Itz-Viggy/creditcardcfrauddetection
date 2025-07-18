# Credit Card Fraud Detection - Phase 1 MVP

## Project Overview
A batch fraud detection system that processes CSV data and displays results on a Streamlit dashboard.

## Features
- ✅ Exploratory Data Analysis (EDA)
- ✅ Data Preprocessing & Feature Engineering
- ✅ Multiple ML Models (Logistic Regression, XGBoost, LightGBM)
- ✅ Interactive Streamlit Dashboard
- ✅ Geographic Heatmap Visualization
- ✅ Model Performance Metrics

## Project Structure
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

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Download dataset: `python download_data.py`
3. Run EDA: Open `notebooks/eda.ipynb`
4. Train models: `python main.py`
5. Launch dashboard: `streamlit run dashboard/app.py`

## Dataset
Using the Kaggle Credit Card Fraud Detection Dataset.
