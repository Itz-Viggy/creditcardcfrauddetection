"""
Feature Engineering for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Advanced feature engineering for fraud detection
    """
    
    def __init__(self):
        self.feature_stats = {}
        self.created_features = []
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical aggregation features
        """
        print("📊 Creating statistical features...")
        
        df_stats = df.copy()
        
        # V features (PCA components) statistical operations
        v_features = [col for col in df.columns if col.startswith('V')]
        
        if v_features:
            # Statistical aggregations across V features
            df_stats['v_mean'] = df_stats[v_features].mean(axis=1)
            df_stats['v_std'] = df_stats[v_features].std(axis=1)
            df_stats['v_min'] = df_stats[v_features].min(axis=1)
            df_stats['v_max'] = df_stats[v_features].max(axis=1)
            df_stats['v_median'] = df_stats[v_features].median(axis=1)
            df_stats['v_range'] = df_stats['v_max'] - df_stats['v_min']
            df_stats['v_skew'] = df_stats[v_features].skew(axis=1)
            df_stats['v_kurtosis'] = df_stats[v_features].kurtosis(axis=1)
            
            # Count of extreme values
            df_stats['v_extreme_count'] = (np.abs(df_stats[v_features]) > 3).sum(axis=1)
            
            # Pairwise feature interactions (select a few important ones)
            if len(v_features) >= 4:
                df_stats['v1_v2_interaction'] = df_stats['V1'] * df_stats['V2']
                df_stats['v3_v4_interaction'] = df_stats['V3'] * df_stats['V4']
                df_stats['v1_v3_ratio'] = np.where(df_stats['V3'] != 0, df_stats['V1'] / df_stats['V3'], 0)
                df_stats['v2_v4_ratio'] = np.where(df_stats['V4'] != 0, df_stats['V2'] / df_stats['V4'], 0)
            
            self.created_features.extend([
                'v_mean', 'v_std', 'v_min', 'v_max', 'v_median', 'v_range', 
                'v_skew', 'v_kurtosis', 'v_extreme_count'
            ])
            
            if len(v_features) >= 4:
                self.created_features.extend([
                    'v1_v2_interaction', 'v3_v4_interaction', 'v1_v3_ratio', 'v2_v4_ratio'
                ])
        
        print(f"✅ Created {len([f for f in self.created_features if f.startswith('v_')])} statistical features")
        
        return df_stats
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive time-based features
        """
        print("⏰ Creating advanced time features...")
        
        df_time = df.copy()
        
        if 'Time' not in df_time.columns:
            print("⚠️  Time column not found, skipping time features")
            return df_time
        
        # Basic time conversions
        df_time['hour'] = (df_time['Time'] / 3600) % 24
        df_time['day'] = (df_time['Time'] / (24 * 3600)).astype(int)
        df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
        df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
        
        # Time period indicators (only add if they have variance)
        weekend = (df_time['day'] % 7).isin([5, 6]).astype(int)
        if weekend.std() > 0:
            df_time['is_weekend'] = weekend
        
        night = ((df_time['hour'] <= 6) | (df_time['hour'] >= 22)).astype(int)
        if night.std() > 0:
            df_time['is_night'] = night
        
        business_hours = ((df_time['hour'] >= 9) & (df_time['hour'] <= 17)).astype(int)
        if business_hours.std() > 0:
            df_time['is_business_hours'] = business_hours
        
        lunch_time = ((df_time['hour'] >= 11) & (df_time['hour'] <= 14)).astype(int)
        if lunch_time.std() > 0:
            df_time['is_lunch_time'] = lunch_time
        
        evening = ((df_time['hour'] >= 17) & (df_time['hour'] <= 21)).astype(int)
        if evening.std() > 0:
            df_time['is_evening'] = evening
        
        # Time since start of dataset
        min_time = df_time['Time'].min()
        df_time['time_from_start'] = df_time['Time'] - min_time
        df_time['days_from_start'] = df_time['time_from_start'] / (24 * 3600)
        
        # Time differences (sorted by time)
        df_sorted = df_time.sort_values('Time').copy()
        df_sorted['time_since_last'] = df_sorted['Time'].diff().fillna(0)
        df_sorted['time_to_next'] = df_sorted['Time'].diff(-1).fillna(0).abs()
        
        # Merge back to original order
        df_time['time_since_last'] = df_sorted.reindex(df_time.index)['time_since_last']
        df_time['time_to_next'] = df_sorted.reindex(df_time.index)['time_to_next']
        
        # Time velocity features
        df_time['transaction_velocity'] = 1 / (df_time['time_since_last'] + 1)  # Add 1 to avoid division by zero
        
        # Collect features that were actually created
        time_features = [
            'hour', 'day', 'hour_sin', 'hour_cos', 'time_from_start',
            'days_from_start', 'time_since_last', 'time_to_next', 'transaction_velocity'
        ]
        
        # Add conditional features that exist
        if 'is_weekend' in df_time.columns:
            time_features.append('is_weekend')
        if 'is_night' in df_time.columns:
            time_features.append('is_night')
        if 'is_business_hours' in df_time.columns:
            time_features.append('is_business_hours')
        if 'is_lunch_time' in df_time.columns:
            time_features.append('is_lunch_time')
        if 'is_evening' in df_time.columns:
            time_features.append('is_evening')
        
        self.created_features.extend(time_features)
        print(f"✅ Created {len(time_features)} time-based features")
        
        return df_time
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive amount-based features
        """
        print("💰 Creating advanced amount features...")
        
        df_amount = df.copy()
        
        if 'Amount' not in df_amount.columns:
            print("⚠️  Amount column not found, skipping amount features")
            return df_amount
        
        # Basic transformations
        df_amount['amount_log'] = np.log1p(df_amount['Amount'])
        df_amount['amount_sqrt'] = np.sqrt(df_amount['Amount'])
        df_amount['amount_squared'] = df_amount['Amount'] ** 2
        
        # Statistical features
        amount_mean = df_amount['Amount'].mean()
        amount_std = df_amount['Amount'].std()
        amount_median = df_amount['Amount'].median()
        
        df_amount['amount_zscore'] = (df_amount['Amount'] - amount_mean) / amount_std
        df_amount['amount_deviation_from_median'] = df_amount['Amount'] - amount_median
        df_amount['amount_percentile'] = df_amount['Amount'].rank(pct=True)
        
        # Amount categories (simplified to avoid zero variance)
        amount_bins = [0, 10, 50, 200, 1000, float('inf')]
        amount_labels = ['micro', 'small', 'medium', 'large', 'very_large']
        df_amount['amount_category'] = pd.cut(df_amount['Amount'], bins=amount_bins, labels=amount_labels)
        
        # One-hot encode categories (only if they have variance)
        for label in amount_labels:
            feature_name = f'amount_{label}'
            df_amount[feature_name] = (df_amount['amount_category'] == label).astype(int)
            # Remove if zero variance
            if df_amount[feature_name].std() == 0:
                df_amount.drop(feature_name, axis=1, inplace=True)
        
        df_amount.drop('amount_category', axis=1, inplace=True)
        
        # Extreme amount indicators (only if they have variance)
        q99 = df_amount['Amount'].quantile(0.99)
        q01 = df_amount['Amount'].quantile(0.01)
        
        df_amount['is_extreme_high'] = (df_amount['Amount'] > q99).astype(int)
        df_amount['is_extreme_low'] = (df_amount['Amount'] < q01).astype(int)
        
        # Only add round amount features if they have variance
        round_amount = (df_amount['Amount'] % 1 == 0).astype(int)
        if round_amount.std() > 0:
            df_amount['is_round_amount'] = round_amount
        
        very_round = (df_amount['Amount'] % 10 == 0).astype(int)
        if very_round.std() > 0:
            df_amount['is_very_round'] = very_round
        
        # Simplified rolling statistics (avoid NaN issues)
        if 'Time' in df_amount.columns:
            df_sorted = df_amount.sort_values('Time')
            
            # Only use smaller windows to avoid NaN issues
            window_sizes = [5, 10]
            for window in window_sizes:
                if len(df_sorted) >= window:
                    rolling_mean = df_sorted['Amount'].rolling(window=window, min_periods=1).mean()
                    rolling_std = df_sorted['Amount'].rolling(window=window, min_periods=1).std()
                    
                    # Fill NaN values with the mean
                    rolling_std = rolling_std.fillna(rolling_std.mean())
                    
                    df_amount[f'amount_rolling_mean_{window}'] = rolling_mean.reindex(df_amount.index)
                    df_amount[f'amount_rolling_std_{window}'] = rolling_std.reindex(df_amount.index)
        
        # Collect features that were actually created
        amount_features = [
            'amount_log', 'amount_sqrt', 'amount_squared', 'amount_zscore',
            'amount_deviation_from_median', 'amount_percentile'
        ]
        
        # Add categorical features that exist
        for label in amount_labels:
            feature_name = f'amount_{label}'
            if feature_name in df_amount.columns:
                amount_features.append(feature_name)
        
        # Add indicator features that exist
        if 'is_extreme_high' in df_amount.columns:
            amount_features.append('is_extreme_high')
        if 'is_extreme_low' in df_amount.columns:
            amount_features.append('is_extreme_low')
        if 'is_round_amount' in df_amount.columns:
            amount_features.append('is_round_amount')
        if 'is_very_round' in df_amount.columns:
            amount_features.append('is_very_round')
        
        # Add rolling features that exist
        rolling_features = [col for col in df_amount.columns if 'rolling' in col]
        amount_features.extend(rolling_features)
        
        self.created_features.extend(amount_features)
        print(f"✅ Created {len(amount_features)} amount-based features")
        
        return df_amount
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different components
        """
        print("🔗 Creating interaction features...")
        
        df_interact = df.copy()
        
        # Amount-Time interactions
        if 'Amount' in df_interact.columns and 'Time' in df_interact.columns:
            df_interact['amount_time_interaction'] = df_interact['Amount'] * df_interact['Time']
            
            if 'hour' in df_interact.columns:
                df_interact['amount_hour_interaction'] = df_interact['Amount'] * df_interact['hour']
            
            if 'is_night' in df_interact.columns:
                df_interact['amount_night_interaction'] = df_interact['Amount'] * df_interact['is_night']
            
            if 'is_weekend' in df_interact.columns:
                df_interact['amount_weekend_interaction'] = df_interact['Amount'] * df_interact['is_weekend']
        
        # V feature interactions with amount
        v_features = [col for col in df_interact.columns if col.startswith('V')]
        if v_features and 'Amount' in df_interact.columns:
            # Select a few key V features for interactions
            key_v_features = v_features[:5]  # Use first 5 V features
            
            for v_feature in key_v_features:
                df_interact[f'{v_feature}_amount_interaction'] = df_interact[v_feature] * df_interact['Amount']
                
                if 'amount_log' in df_interact.columns:
                    df_interact[f'{v_feature}_amount_log_interaction'] = df_interact[v_feature] * df_interact['amount_log']
        
        # High-level feature combinations
        if 'v_mean' in df_interact.columns and 'Amount' in df_interact.columns:
            df_interact['v_mean_amount_ratio'] = np.where(
                df_interact['Amount'] != 0, 
                df_interact['v_mean'] / df_interact['Amount'], 
                0
            )
        
        interaction_features = [col for col in df_interact.columns if 'interaction' in col or '_ratio' in col]
        self.created_features.extend([f for f in interaction_features if f not in self.created_features])
        
        print(f"✅ Created {len(interaction_features)} interaction features")
        
        return df_interact
    
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific risk indicators
        """
        print("⚠️  Creating risk indicator features...")
        
        df_risk = df.copy()
        
        # Transaction risk scores based on domain knowledge
        risk_features = []
        
        # High-risk time periods
        if 'hour' in df_risk.columns:
            df_risk['high_risk_hour'] = ((df_risk['hour'] >= 2) & (df_risk['hour'] <= 5)).astype(int)
            risk_features.append('high_risk_hour')
        
        # High-risk amounts
        if 'Amount' in df_risk.columns:
            # Very small or very large amounts
            q95 = df_risk['Amount'].quantile(0.95)
            df_risk['high_risk_amount'] = ((df_risk['Amount'] < 1) | (df_risk['Amount'] > q95)).astype(int)
            risk_features.append('high_risk_amount')
        
        # Suspicious patterns in V features
        v_features = [col for col in df_risk.columns if col.startswith('V')]
        if len(v_features) >= 10:
            # Look for unusual patterns in PCA components
            df_risk['suspicious_v_pattern'] = 0
            
            # Multiple extreme values
            extreme_threshold = 3
            extreme_count = (np.abs(df_risk[v_features[:10]]) > extreme_threshold).sum(axis=1)
            df_risk['suspicious_v_pattern'] += (extreme_count >= 3).astype(int)
            
            # All values close to zero (unusual for real transactions)
            near_zero_count = (np.abs(df_risk[v_features[:10]]) < 0.1).sum(axis=1)
            df_risk['suspicious_v_pattern'] += (near_zero_count >= 8).astype(int)
            
            risk_features.append('suspicious_v_pattern')
        
        # Combined risk score
        if risk_features:
            df_risk['total_risk_score'] = df_risk[risk_features].sum(axis=1)
            risk_features.append('total_risk_score')
        
        self.created_features.extend(risk_features)
        print(f"✅ Created {len(risk_features)} risk indicator features")
        
        return df_risk
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        """
        print("🚀 Starting feature engineering pipeline...")
        
        # Reset created features list
        self.created_features = []
        
        # Apply all feature engineering steps
        df_engineered = df.copy()
        
        # 1. Statistical features
        df_engineered = self.create_statistical_features(df_engineered)
        
        # 2. Time features
        df_engineered = self.create_time_features(df_engineered)
        
        # 3. Amount features
        df_engineered = self.create_amount_features(df_engineered)
        
        # 4. Interaction features
        df_engineered = self.create_interaction_features(df_engineered)
        
        # 5. Risk features
        df_engineered = self.create_risk_features(df_engineered)
        
        # Store feature statistics
        self.feature_stats = {
            'original_features': len(df.columns),
            'created_features': len(self.created_features),
            'total_features': len(df_engineered.columns),
            'feature_list': self.created_features
        }
        
        print(f"\n✅ Feature engineering completed!")
        print(f"📊 Original features: {self.feature_stats['original_features']}")
        print(f"📊 Created features: {self.feature_stats['created_features']}")
        print(f"📊 Total features: {self.feature_stats['total_features']}")
        print(f"📊 Feature increase: {self.feature_stats['total_features'] - self.feature_stats['original_features']} (+{((self.feature_stats['total_features'] / self.feature_stats['original_features']) - 1) * 100:.1f}%)")
        
        return df_engineered
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Group features by type for analysis
        """
        groups = {
            'original': [col for col in self.created_features if not any(prefix in col for prefix in ['v_', 'hour', 'day', 'amount_', 'time_', 'is_', 'rolling', 'interaction', 'risk'])],
            'statistical': [col for col in self.created_features if col.startswith('v_')],
            'time': [col for col in self.created_features if any(prefix in col for prefix in ['hour', 'day', 'time_', 'is_'])],
            'amount': [col for col in self.created_features if col.startswith('amount_') or 'rolling' in col],
            'interaction': [col for col in self.created_features if 'interaction' in col or '_ratio' in col],
            'risk': [col for col in self.created_features if any(prefix in col for prefix in ['risk', 'suspicious', 'extreme'])]
        }
        
        return groups
