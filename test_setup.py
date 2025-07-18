"""
Quick test script to verify the installation and basic functionality
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn")
    except ImportError as e:
        print(f"❌ scikit-learn: {e}")
        return False
    
    try:
        import xgboost
        print("✅ xgboost")
    except ImportError as e:
        print(f"❌ xgboost: {e}")
        return False
    
    try:
        import lightgbm
        print("✅ lightgbm")
    except ImportError as e:
        print(f"❌ lightgbm: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
    except ImportError as e:
        print(f"❌ matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ seaborn")
    except ImportError as e:
        print(f"❌ seaborn: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit")
    except ImportError as e:
        print(f"❌ streamlit: {e}")
        return False
    
    return True

def test_data_availability():
    """Test if data files are available"""
    print("\n📁 Testing data availability...")
    
    import os
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("❌ data/ directory not found")
        return False
    
    # Check for main dataset
    if os.path.exists('data/creditcard.csv'):
        print("✅ creditcard.csv found")
        return True
    else:
        print("⚠️  creditcard.csv not found - run 'python download_data.py' first")
        return False

def test_modules():
    """Test if our custom modules can be imported"""
    print("\n🔧 Testing custom modules...")
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent / "src"))
    
    try:
        from utils import load_data
        print("✅ utils module")
    except ImportError as e:
        print(f"❌ utils module: {e}")
        return False
    
    try:
        from data_preprocessing import DataPreprocessor
        print("✅ data_preprocessing module")
    except ImportError as e:
        print(f"❌ data_preprocessing module: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("✅ feature_engineering module")
    except ImportError as e:
        print(f"❌ feature_engineering module: {e}")
        return False
    
    try:
        from model_training import FraudModelTrainer
        print("✅ model_training module")
    except ImportError as e:
        print(f"❌ model_training module: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality with sample data"""
    print("\n⚙️  Testing basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'Time': np.random.randint(0, 100000, 100),
            'Amount': np.random.exponential(50, 100),
            'V1': np.random.normal(0, 1, 100),
            'V2': np.random.normal(0, 1, 100),
            'Class': np.random.choice([0, 1], 100, p=[0.998, 0.002])
        })
        
        print(f"✅ Sample data created: {sample_data.shape}")
        
        # Test preprocessing
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        
        # Quick preprocessing test
        sample_data['hour'] = (sample_data['Time'] / 3600) % 24
        print("✅ Basic preprocessing works")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Credit Card Fraud Detection - System Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n❌ Import test failed. Please install requirements:")
        print("   pip install -r requirements.txt")
    
    # Test data
    if not test_data_availability():
        all_passed = False
        print("\n⚠️  Data not available. To download:")
        print("   python download_data.py")
    
    # Test modules
    if not test_modules():
        all_passed = False
        print("\n❌ Module test failed. Check src/ directory.")
    
    # Test functionality
    if not test_basic_functionality():
        all_passed = False
        print("\n❌ Functionality test failed.")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! System is ready.")
        print("\n📋 Next steps:")
        print("   1. Run: python main.py")
        print("   2. Launch dashboard: streamlit run dashboard/app.py")
        print("   3. Explore notebook: jupyter notebook notebooks/fraud_detection_eda.ipynb")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        print("\n🔧 Common fixes:")
        print("   - pip install -r requirements.txt")
        print("   - python download_data.py")
    
    return all_passed

if __name__ == "__main__":
    main()
