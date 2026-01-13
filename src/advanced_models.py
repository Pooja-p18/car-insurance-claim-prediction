"""
Advanced Models for Car Insurance Claim Prediction
===================================================
Random Forest, XGBoost, and LightGBM
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


class AdvancedModels:
    """Train and evaluate advanced ensemble models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("TRAINING: RANDOM FOREST")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test,
            "Random Forest"
        )
        
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        return results
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting model"""
        print("\n" + "="*60)
        print("TRAINING: GRADIENT BOOSTING")
        print("="*60)
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state,
            verbose=0
        )
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test,
            "Gradient Boosting"
        )
        
        self.models['gradient_boosting'] = model
        self.results['gradient_boosting'] = results
        
        return results
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING: XGBOOST")
        print("="*60)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            verbosity=0
        )
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test,
            "XGBoost"
        )
        
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        return results
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model"""
        print("\n" + "="*60)
        print("TRAINING: LIGHTGBM")
        print("="*60)
        
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=self.random_state,
            verbose=-1
        )
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test,
            "LightGBM"
        )
        
        self.models['lightgbm'] = model
        self.results['lightgbm'] = results
        
        return results
    
    def tune_hyperparameters(self, model_name, X_train, y_train):
        """
        Tune hyperparameters using GridSearchCV
        
        Args:
            model_name: 'random_forest', 'xgboost', or 'lightgbm'
            X_train, y_train: Training data
            
        Returns:
            Best model after tuning
        """
        print("\n" + "="*60)
        print(f"HYPERPARAMETER TUNING: {model_name.upper()}")
        print("="*60)
        
        if model_name == 'random_forest':
            model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [5, 10]
            }
        
        elif model_name == 'xgboost':
            model = XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            )
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 6, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        
        elif model_name == 'lightgbm':
            model = LGBMClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            )
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 6, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'num_leaves': [31, 50]
            }
        else:
            print(f"Model {model_name} not supported for tuning")
            return None
        
        # GridSearch with cross-validation
        print("Starting grid search (this may take a while)...")
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n✓ Grid search complete!")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """Evaluate model performance"""
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred_test)
        
        # Print results
        print(f"\n{'─'*60}")
        print(f"RESULTS: {model_name}")
        print(f"{'─'*60}")
        print(f"Train Accuracy:  {train_acc:.4f}")
        print(f"Test Accuracy:   {test_acc:.4f}")
        print(f"Precision:       {precision:.4f}")
        print(f"Recall:          {recall:.4f}")
        print(f"F1-Score:        {f1:.4f}")
        print(f"ROC-AUC:         {roc_auc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 No    Yes")
        print(f"Actual No     {cm[0][0]:5d} {cm[0][1]:5d}")
        print(f"       Yes    {cm[1][0]:5d} {cm[1][1]:5d}")
        
        # Check overfitting
        overfit_diff = train_acc - test_acc
        if overfit_diff > 0.05:
            print(f"\n⚠️  Warning: Possible overfitting detected")
            print(f"   Train-Test gap: {overfit_diff:.4f}")
        else:
            print(f"\n✓ Good generalization (Train-Test gap: {overfit_diff:.4f})")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_5_idx = np.argsort(importances)[-5:][::-1]
            print(f"\nTop 5 Important Features:")
            for idx in top_5_idx:
                print(f"  Feature {idx}: {importances[idx]:.4f}")
        
        return {
            'model_name': model_name,
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred_proba': y_pred_proba
        }
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.results:
            print("No models trained yet!")
            return None
        
        best_name = max(self.results, key=lambda k: self.results[k]['roc_auc'])
        best_results = self.results[best_name]
        
        print(f"\n🏆 Best Advanced Model: {best_results['model_name']}")
        print(f"   ROC-AUC: {best_results['roc_auc']:.4f}")
        
        return best_name, best_results


if __name__ == "__main__":
    """Test advanced models"""
    import pandas as pd
    
    print("Testing Advanced Models...")
    
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    print(f"Data loaded: Train={X_train.shape}, Test={X_test.shape}")
    
    # Create advanced models
    advanced = AdvancedModels(random_state=42)
    
    # Train all models
    rf_results = advanced.train_random_forest(X_train, y_train, X_test, y_test)
    gb_results = advanced.train_gradient_boosting(X_train, y_train, X_test, y_test)
    xgb_results = advanced.train_xgboost(X_train, y_train, X_test, y_test)
    lgbm_results = advanced.train_lightgbm(X_train, y_train, X_test, y_test)
    
    # Get best model
    best_name, best_results = advanced.get_best_model()
    
    print("\n✓ Advanced models test successful!")