"""
Model Training for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class FraudModelTrainer:
    """
    Train and evaluate multiple models for fraud detection
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.model_performances = {}
        self.best_model = None
        self.best_score = 0
        
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """
        Train Logistic Regression baseline model
        """
        print("🎯 Training Logistic Regression...")
        
        # Debug: Check input data
        print(f"   X_train shape: {X_train.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_train unique values: {y_train.unique()}")
        print(f"   y_train value counts: {y_train.value_counts().to_dict()}")
        
        # Check for data issues
        if X_train.isnull().sum().sum() > 0:
            print("   ⚠️  X_train contains NaN values")
        if np.isinf(X_train.values).sum() > 0:
            print("   ⚠️  X_train contains infinite values")
        
        # Use class_weight='balanced' to handle imbalance
        model = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear'  # Good for binary classification
        )
        
        # Train model
        try:
            model.fit(X_train, y_train)
            print("   ✅ Model training completed")
        except Exception as e:
            print(f"   ❌ Model training failed: {e}")
            raise e
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        print(f"   Predictions shape: {y_pred.shape}")
        print(f"   Probabilities shape: {y_prob.shape}")
        print(f"   y_pred unique values: {np.unique(y_pred)}")
        print(f"   y_prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
        
        # Evaluate
        performance = self._evaluate_model(y_test, y_pred, y_prob, "Logistic Regression")
        
        # Store model and performance
        self.models['logistic_regression'] = model
        self.model_performances['logistic_regression'] = performance
        
        return model, performance
    

    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost model
        """
        print("🚀 Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced dataset
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Train model
        model.fit(X_train, y_train, 
                 eval_set=[(X_test, y_test)], 
                 early_stopping_rounds=10,
                 verbose=False)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        performance = self._evaluate_model(y_test, y_pred, y_prob, "XGBoost")
        
        # Store model and performance
        self.models['xgboost'] = model
        self.model_performances['xgboost'] = performance
        
        return model, performance
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """
        Train LightGBM model
        """
        print("💡 Training LightGBM...")
        
        # Calculate scale_pos_weight for imbalanced dataset
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=self.random_state,
            verbose=-1
        )
        
        # Train model
        model.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 early_stopping_rounds=10,
                 verbose=False)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        performance = self._evaluate_model(y_test, y_pred, y_prob, "LightGBM")
        
        # Store model and performance
        self.models['lightgbm'] = model
        self.model_performances['lightgbm'] = performance
        
        return model, performance
    
    def _evaluate_model(self, y_true, y_pred, y_prob, model_name):
        """
        Comprehensive model evaluation
        """
        print(f"\n📊 Evaluating {model_name}...")
        
        # Basic metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUC scores
        roc_auc = roc_auc_score(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        # Calculate Recall at different FPR thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Find recall at 1% and 5% FPR
        recall_at_1_fpr = self._recall_at_fpr(fpr, tpr, 0.01)
        recall_at_5_fpr = self._recall_at_fpr(fpr, tpr, 0.05)
        
        performance = {
            'model_name': model_name,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'recall_at_1_fpr': recall_at_1_fpr,
            'recall_at_5_fpr': recall_at_5_fpr,
            'confusion_matrix': cm,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        # Print results
        print(f"✅ {model_name} Results:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}")
        print(f"   Avg Precision: {avg_precision:.4f}")
        print(f"   Recall @ 1% FPR: {recall_at_1_fpr:.4f}")
        print(f"   Recall @ 5% FPR: {recall_at_5_fpr:.4f}")
        
        return performance
    
    def _recall_at_fpr(self, fpr, tpr, target_fpr):
        """
        Calculate recall at specific FPR threshold
        """
        # Find the closest FPR to target
        idx = np.argmin(np.abs(fpr - target_fpr))
        return tpr[idx]
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """
        Perform cross-validation on all trained models
        """
        print(f"\n🔄 Performing {cv_folds}-fold cross-validation...")
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, model in self.models.items():
            print(f"   Cross-validating {model_name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
            
            cv_results[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"      ROC AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return cv_results
    
    def get_feature_importance(self, model_name, feature_names, top_n=20):
        """
        Get feature importance for tree-based models
        """
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_[0])
        else:
            print(f"❌ Model {model_name} doesn't support feature importance")
            return None
        
        # Create DataFrame
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_imp_df.head(top_n)
    
    def select_best_model(self, metric='avg_precision'):
        """
        Select the best model based on specified metric
        """
        print(f"\n🏆 Selecting best model based on {metric}...")
        
        best_score = 0
        best_model_name = None
        
        for model_name, performance in self.model_performances.items():
            score = performance.get(metric, 0)
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name:
            self.best_model = self.models[best_model_name]
            self.best_score = best_score
            
            print(f"✅ Best model: {best_model_name}")
            print(f"   {metric}: {best_score:.4f}")
            
            return best_model_name, self.best_model
        else:
            print("❌ No models found")
            return None, None
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all available models
        """
        print("🚀 Training all models...")
        
        results = {}
        
        # Train each model
        try:
            model, perf = self.train_logistic_regression(X_train, y_train, X_test, y_test)
            results['logistic_regression'] = (model, perf)
        except Exception as e:
            print(f"❌ Error training Logistic Regression: {e}")
        
        try:
            model, perf = self.train_xgboost(X_train, y_train, X_test, y_test)
            results['xgboost'] = (model, perf)
        except Exception as e:
            print(f"❌ Error training XGBoost: {e}")
        
        try:
            model, perf = self.train_lightgbm(X_train, y_train, X_test, y_test)
            results['lightgbm'] = (model, perf)
        except Exception as e:
            print(f"❌ Error training LightGBM: {e}")
        
        # Select best model
        if self.model_performances:
            self.select_best_model()
        
        return results
    
    def save_models(self, save_dir="models"):
        """
        Save all trained models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = f"{save_dir}/{model_name}_model.joblib"
            joblib.dump(model, model_path)
            print(f"✅ Saved {model_name} to {model_path}")
        
        # Save performances
        if self.model_performances:
            perf_path = f"{save_dir}/model_performances.joblib"
            joblib.dump(self.model_performances, perf_path)
            print(f"✅ Saved performances to {perf_path}")
    
    def load_models(self, save_dir="models"):
        """
        Load saved models
        """
        import os
        
        model_files = {
            'logistic_regression': f"{save_dir}/logistic_regression_model.joblib",
            'xgboost': f"{save_dir}/xgboost_model.joblib",
            'lightgbm': f"{save_dir}/lightgbm_model.joblib"
        }
        
        for model_name, model_path in model_files.items():
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"✅ Loaded {model_name} from {model_path}")
        
        # Load performances
        perf_path = f"{save_dir}/model_performances.joblib"
        if os.path.exists(perf_path):
            self.model_performances = joblib.load(perf_path)
            print(f"✅ Loaded performances from {perf_path}")
    
    def predict_fraud_probability(self, X, model_name=None):
        """
        Predict fraud probability using specified model or best model
        """
        if model_name is None:
            if self.best_model is not None:
                model = self.best_model
                model_name = "best_model"
            else:
                print("❌ No model specified and no best model selected")
                return None
        else:
            if model_name not in self.models:
                print(f"❌ Model {model_name} not found")
                return None
            model = self.models[model_name]
        
        try:
            probabilities = model.predict_proba(X)[:, 1]
            predictions = model.predict(X)
            
            return {
                'probabilities': probabilities,
                'predictions': predictions,
                'model_used': model_name
            }
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return None
