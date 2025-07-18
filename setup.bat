@echo off
echo ============================================================
echo Credit Card Fraud Detection - Phase 1 MVP Setup
echo ============================================================
echo.

echo 🚀 Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo 📊 Testing setup...
python test_setup.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Setup test failed
    pause
    exit /b 1
)

echo.
echo 📁 Downloading dataset...
python download_data.py
if %ERRORLEVEL% neq 0 (
    echo ⚠️  Dataset download failed - you can download manually
    echo Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    echo Download creditcard.csv to the data/ folder
)

echo.
echo 🤖 Running fraud detection pipeline...
python main.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Pipeline failed
    pause
    exit /b 1
)

echo.
echo ✅ Setup completed successfully!
echo.
echo 📋 What's been created:
echo   📊 Trained models in models/ directory
echo   📈 Analysis results in data/ directory  
echo   🗒️  Jupyter notebook ready to run
echo   📱 Dashboard ready to launch
echo.
echo 🎯 Next steps:
echo   1. Launch dashboard: streamlit run dashboard/app.py
echo   2. Explore notebook: jupyter notebook notebooks/fraud_detection_eda.ipynb
echo   3. Review results in models/ and data/ directories
echo.
pause
