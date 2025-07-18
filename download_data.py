"""
Data Download Script for Credit Card Fraud Detection
Downloads the Kaggle Credit Card Fraud Detection dataset
"""

import os
import zipfile
import pandas as pd
from pathlib import Path

def download_kaggle_dataset():
    """
    Download the Credit Card Fraud Detection dataset from Kaggle
    
    Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Set up Kaggle API credentials:
       - Go to Kaggle Account settings
       - Create new API token
       - Place kaggle.json in ~/.kaggle/ (or C:\\Users\\{username}\\.kaggle\\ on Windows)
    """
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if dataset already exists
    if check_existing_dataset():
        print("✅ Using existing dataset")
        return
    
    # Check if Kaggle credentials are configured
    # Check both WSL and Windows paths
    kaggle_config_paths = [
        Path.home() / ".kaggle" / "kaggle.json",  # WSL/Linux path
        Path("/mnt/c/Users") / os.getenv('USER', 'vigne') / ".kaggle" / "kaggle.json",  # Windows path via WSL
        Path("C:/Users/vigne/.kaggle/kaggle.json")  # Direct Windows path
    ]
    
    kaggle_config_path = None
    for path in kaggle_config_paths:
        if path.exists():
            kaggle_config_path = path
            break
    
    if kaggle_config_path is None:
        print("⚠️  Kaggle credentials not found!")
        print(f"Checked paths: {[str(p) for p in kaggle_config_paths]}")
        print("\n🔧 To set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Place the downloaded kaggle.json file in ~/.kaggle/ folder")
        print("\nFalling back to sample dataset creation...")
        create_sample_dataset()
        return
    
    print(f"✅ Found Kaggle credentials at: {kaggle_config_path}")
    
    # Set environment variable for Kaggle API
    os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_config_path.parent)
    
    try:
        print("📥 Attempting to download dataset from Kaggle...")
        
        # Use subprocess for better error handling
        import subprocess
        result = subprocess.run([
            "kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", 
            "-p", "data", "--unzip"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Kaggle download failed: {result.stderr}")
            print("Falling back to sample dataset creation...")
            create_sample_dataset()
            return
        
        print("✅ Dataset downloaded successfully!")
        
        # Verify the download
        csv_file = data_dir / "creditcard.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"📊 Dataset shape: {df.shape}")
            print(f"📊 Columns: {list(df.columns)}")
            print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Quick peek at the data
            print("\n📈 Dataset preview:")
            print(df.head())
            
            # Check for fraud distribution
            fraud_count = df['Class'].value_counts()
            print(f"\n🎯 Fraud distribution:")
            print(f"Normal transactions: {fraud_count[0]:,} ({fraud_count[0]/len(df)*100:.2f}%)")
            print(f"Fraudulent transactions: {fraud_count[1]:,} ({fraud_count[1]/len(df)*100:.2f}%)")
            
        else:
            print("❌ CSV file not found after download")
            print("Creating sample dataset instead...")
            create_sample_dataset()
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n🔧 Manual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. Click 'Download' button")
        print("3. Extract creditcard.csv to the 'data' folder")
        print("\nCreating sample dataset for now...")
        
        # Create a sample dataset for development if download fails
        create_sample_dataset()

def create_sample_dataset():
    """Create a small sample dataset for development purposes"""
    import numpy as np
    
    print("🔧 Creating sample dataset for development...")
    
    # Create a small sample dataset with similar structure
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features (V1-V28 are PCA components in real dataset)
    data = {
        'Time': np.random.randint(0, 172800, n_samples),  # Up to 48 hours
        'Amount': np.random.exponential(50, n_samples),   # Exponential distribution for amounts
    }
    
    # Add V1-V28 features (normally distributed like PCA components)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Create imbalanced target (similar to real dataset)
    fraud_rate = 0.05  # 5% fraud rate (increased for better training)
    n_fraud = int(n_samples * fraud_rate)
    data['Class'] = np.concatenate([
        np.zeros(n_samples - n_fraud),
        np.ones(n_fraud)
    ])
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    for key in data:
        data[key] = data[key][indices]
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv("data/creditcard.csv", index=False)
    
    print(f"✅ Sample dataset created with {n_samples} transactions")
    print(f"📊 Fraud rate: {fraud_rate*100:.1f}%")

def check_existing_dataset():
    """Check if the dataset already exists locally"""
    csv_file = Path("data/creditcard.csv")
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ Found existing dataset: {df.shape}")
            
            # Quick validation
            expected_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
            if all(col in df.columns for col in expected_columns):
                print("✅ Dataset appears to be valid")
                
                # Show basic stats
                fraud_count = df['Class'].value_counts()
                print(f"📊 Normal transactions: {fraud_count[0]:,}")
                print(f"📊 Fraudulent transactions: {fraud_count[1]:,}")
                return True
            else:
                print("⚠️  Dataset structure doesn't match expected format")
                return False
        except Exception as e:
            print(f"⚠️  Error reading existing dataset: {e}")
            return False
    return False

if __name__ == "__main__":
    print("🚀 Credit Card Fraud Detection Dataset Setup")
    print("=" * 50)
    
    # First check if dataset already exists
    if check_existing_dataset():
        print("✅ Dataset already available - no download needed!")
        print("\n🎯 Ready to proceed with:")
        print("   python main.py")
        print("   streamlit run dashboard/app.py")
    else:
        print("📥 Dataset not found locally - attempting download...")
        download_kaggle_dataset()
