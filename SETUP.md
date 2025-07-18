# Installation and Setup Guide

## Quick Start

1. **Clone/Download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download dataset**:
   ```bash
   python download_data.py
   ```
4. **Run the complete pipeline**:
   ```bash
   python main.py
   ```
5. **Launch dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

## Detailed Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step-by-Step Installation

1. **Create virtual environment** (recommended):
   ```bash
   python -m venv fraud_detection_env
   
   # On Windows:
   fraud_detection_env\Scripts\activate
   
   # On macOS/Linux:
   source fraud_detection_env/bin/activate
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Kaggle API** (for dataset download):
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Click "Create New API Token"
   - Place the downloaded `kaggle.json` file in:
     - Windows: `C:\Users\{username}\.kaggle\`
     - macOS/Linux: `~/.kaggle/`
   
   OR manually download the dataset:
   - Go to [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv` to the `data/` folder

4. **Run the project**:
   ```bash
   # Option 1: Complete pipeline
   python main.py
   
   # Option 2: Interactive analysis
   jupyter notebook notebooks/fraud_detection_eda.ipynb
   
   # Option 3: Just the dashboard (after running main.py)
   streamlit run dashboard/app.py
   ```

## Project Structure

```
CreditCardFraud/
├── data/                   # Data files
│   ├── creditcard.csv      # Raw dataset (downloaded)
│   ├── fraud_predictions.csv
│   └── sample_processed_data.csv
├── src/                    # Source code modules
│   ├── utils.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
├── models/                 # Trained models and results
│   ├── *_model.joblib
│   ├── model_comparison.csv
│   └── feature_importance.csv
├── notebooks/              # Jupyter notebooks
│   └── fraud_detection_eda.ipynb
├── dashboard/              # Streamlit dashboard
│   └── app.py
├── main.py                # Main execution script
├── download_data.py       # Dataset download script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all packages are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset not found**: Download the dataset first
   ```bash
   python download_data.py
   ```

3. **Kaggle API issues**: 
   - Check `kaggle.json` is in the right location
   - Try manual download from Kaggle website

4. **Memory issues**: The dataset is ~150MB, ensure sufficient RAM

5. **Streamlit port conflicts**: 
   ```bash
   streamlit run dashboard/app.py --port 8502
   ```

### Performance Notes

- **Training time**: 2-5 minutes on a modern laptop
- **Memory usage**: ~2GB RAM recommended
- **Storage**: ~500MB for all files

## Features

### Models Implemented
- Logistic Regression (baseline)
- XGBoost
- LightGBM  
- Random Forest

### Evaluation Metrics
- Precision, Recall, F1-Score
- ROC AUC, Average Precision
- Recall at 1% and 5% False Positive Rate
- Confusion Matrix

### Dashboard Features
- Model performance comparison
- Feature importance analysis
- Geographic fraud visualization
- Time-based pattern analysis
- Top fraud cases investigation

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational purposes.
