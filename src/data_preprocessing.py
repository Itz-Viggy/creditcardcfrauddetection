"""
Data Preprocessing for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing for fraud detection
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
        self.target_column = 'Class'
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning operations
        """
        print("🧹 Starting data cleaning...")
        
        df_clean = df.copy()
        initial_shape = df_clean.shape
        
        # Remove duplicates
        duplicates_before = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = duplicates_before - df_clean.duplicated().sum()
        
        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        
        # For this dataset, we typically don't have missing values
        # But let's handle them just in case
        if missing_before > 0:
            # Fill numerical columns with median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        missing_after = df_clean.isnull().sum().sum()
        
        print(f"✅ Data cleaning completed:")
        print(f"   Shape: {initial_shape} → {df_clean.shape}")
        print(f"   Duplicates removed: {duplicates_removed}")
        print(f"   Missing values: {missing_before} → {missing_after}")
        
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect outliers using IQR method
        Note: For fraud detection, outliers might be important signals
        """
        print(f"🔍 Detecting outliers using {method.upper()} method...")
        
        df_outliers = df.copy()
        outlier_counts = {}
        
        # Focus on Amount and Time as they are interpretable
        # V1-V28 are PCA components, outliers there might be fraud signals
        numeric_cols = ['Time', 'Amount']
        
        for col in numeric_cols:
            if col in df_outliers.columns:
                Q1 = df_outliers[col].quantile(0.25)
                Q3 = df_outliers[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df_outliers[col] < lower_bound) | (df_outliers[col] > upper_bound)
                outlier_counts[col] = outliers.sum()
                
                # Mark outliers but don't remove them (they might be fraud)
                df_outliers[f'{col}_outlier'] = outliers
        
        print(f"📊 Outliers detected:")
        for col, count in outlier_counts.items():
            percentage = (count / len(df_outliers)) * 100
            print(f"   {col}: {count} ({percentage:.2f}%)")
        
        return df_outliers
    
    def feature_scaling(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                       method: str = 'robust') -> tuple:
        """
        Scale features using specified method
        Robust scaling is preferred for fraud detection due to outliers
        """
        print(f"⚖️  Applying {method} scaling...")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard' or 'robust'")
        
        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                             method: str = 'smote', random_state: int = 42) -> tuple:
        """
        Handle class imbalance using various techniques
        """
        print(f"⚖️  Handling class imbalance using {method.upper()}...")
        
        initial_counts = y.value_counts()
        print(f"   Original distribution: {dict(initial_counts)}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=random_state, k_neighbors=5)
        elif method == 'undersampling':
            sampler = RandomUnderSampler(random_state=random_state)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=random_state)
        else:
            raise ValueError("Method must be 'smote', 'undersampling', or 'smote_tomek'")
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Convert back to DataFrame/Series
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)
            
            final_counts = y_resampled.value_counts()
            print(f"   Final distribution: {dict(final_counts)}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"⚠️  Could not apply {method}: {e}")
            print("   Returning original data...")
            return X, y
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                    random_state: int = 42, scale_features: bool = True,
                    handle_imbalance: str = None) -> dict:
        """
        Complete data preparation pipeline
        """
        print("🚀 Starting data preparation pipeline...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Detect outliers (for information only)
        df_with_outliers = self.detect_outliers(df_clean)
        
        # Separate features and target
        if self.target_column not in df_clean.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Store feature columns (only numeric ones for modeling)
        all_columns = [col for col in df_clean.columns if col != self.target_column]
        numeric_columns = df_clean[all_columns].select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_columns = [col for col in all_columns if col not in numeric_columns]
        
        if non_numeric_columns:
            print(f"⚠️  Excluding non-numeric columns from modeling: {non_numeric_columns}")
        
        self.feature_columns = numeric_columns
        
        X = df_clean[self.feature_columns]
        y = df_clean[self.target_column]
        
        print(f"📊 Features: {len(self.feature_columns)}")
        print(f"📊 Samples: {len(X)}")
        print(f"📊 Fraud rate: {(y == 1).mean() * 100:.2f}%")
        
        # Split data with fraud-aware strategy
        fraud_count = (y == 1).sum()
        min_fraud_for_stratify = max(2, int(test_size * len(y) * 0.01))  # Need at least 2 fraud cases
        
        if fraud_count < min_fraud_for_stratify:
            print(f"⚠️  Few fraud cases ({fraud_count}), using manual split strategy...")
            # Manual split to ensure fraud cases in both sets
            fraud_indices = y[y == 1].index
            normal_indices = y[y == 0].index
            
            # Ensure at least 1 fraud case in test set
            n_fraud_test = max(1, int(fraud_count * test_size))
            n_normal_test = int(len(normal_indices) * test_size)
            
            # Randomly select test indices
            test_fraud_indices = fraud_indices[:n_fraud_test]
            test_normal_indices = normal_indices[:n_normal_test]
            test_indices = list(test_fraud_indices) + list(test_normal_indices)
            
            train_indices = [idx for idx in y.index if idx not in test_indices]
            
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        else:
            # Standard stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y  # Maintain class distribution
            )
        
        print(f"📊 Train set: {X_train.shape[0]} samples (fraud: {y_train.sum()})")
        print(f"📊 Test set: {X_test.shape[0]} samples (fraud: {y_test.sum()})")
        
        # Scale features
        if scale_features:
            X_train, X_test = self.feature_scaling(X_train, X_test)
        
        # Handle class imbalance (only on training set)
        if handle_imbalance:
            X_train, y_train = self.handle_class_imbalance(X_train, y_train, handle_imbalance)
        
        print("✅ Data preparation completed!")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'original_data': df_clean,
            'outlier_info': df_with_outliers
        }
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor
        """
        if self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call prepare_data() first.")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not defined. Call prepare_data() first.")
        
        # Select and order features
        X = df[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        return X_scaled

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional time-based features
    """
    print("⏰ Creating time-based features...")
    
    df_time = df.copy()
    
    # Convert Time to hours, days, etc.
    df_time['hour'] = (df_time['Time'] / 3600) % 24
    df_time['day_of_transaction'] = df_time['Time'] // (24 * 3600)
    
    # Create time period features
    df_time['is_weekend'] = ((df_time['Time'] // (24 * 3600)) % 7).isin([5, 6]).astype(int)
    df_time['is_night'] = ((df_time['hour'] >= 22) | (df_time['hour'] <= 6)).astype(int)
    df_time['is_business_hours'] = ((df_time['hour'] >= 9) & (df_time['hour'] <= 17)).astype(int)
    
    # Time since last transaction (requires sorting by card, but we don't have card ID)
    # For now, create a general time difference feature
    df_time_sorted = df_time.sort_values('Time')
    df_time_sorted['time_diff'] = df_time_sorted['Time'].diff().fillna(0)
    
    # Merge back to original order
    df_time['time_diff'] = df_time_sorted.reindex(df_time.index)['time_diff']
    
    print(f"✅ Added time-based features:")
    print(f"   - hour, day_of_transaction")
    print(f"   - is_weekend, is_night, is_business_hours")
    print(f"   - time_diff")
    
    return df_time

def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional amount-based features
    """
    print("💰 Creating amount-based features...")
    
    df_amount = df.copy()
    
    # Log transform (add 1 to handle 0 amounts)
    df_amount['amount_log'] = np.log1p(df_amount['Amount'])
    
    # Amount categories
    df_amount['amount_category'] = pd.cut(
        df_amount['Amount'], 
        bins=[0, 10, 50, 200, 1000, float('inf')],
        labels=['micro', 'small', 'medium', 'large', 'very_large']
    )
    
    # Convert categories to dummy variables
    amount_dummies = pd.get_dummies(df_amount['amount_category'], prefix='amount')
    df_amount = pd.concat([df_amount, amount_dummies], axis=1)
    df_amount.drop('amount_category', axis=1, inplace=True)
    
    # Statistical features (rolling statistics would require card grouping)
    # For now, create percentile-based features
    df_amount['amount_percentile'] = df_amount['Amount'].rank(pct=True)
    df_amount['amount_zscore'] = (df_amount['Amount'] - df_amount['Amount'].mean()) / df_amount['Amount'].std()
    
    print(f"✅ Added amount-based features:")
    print(f"   - amount_log, amount_percentile, amount_zscore")
    print(f"   - amount category dummies")
    
    return df_amount
