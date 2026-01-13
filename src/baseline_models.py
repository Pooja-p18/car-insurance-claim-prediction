"""
Baseline Models for Car Insurance Claim Prediction
===================================================
Logistic Regression and Decision Tree
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


class BaselineModels:
    """Train and evaluate baseline models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """
        Train Logistic Regression model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with model and results
        """
        print("\n" + "="*60)
        print("TRAINING: LOGISTIC REGRESSION")
        print("="*60)
        
        # Create model with class balancing
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced',  # Handle class imbalance
            solver='lbfgs'
        )
        
        # Train
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test, 
            "Logistic Regression"
        )
        
        # Store
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        return results
    
    def train_decision_tree(self, X_train, y_train, X_test, y_test):
        """
        Train Decision Tree model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with model and results
        """
        print("\n" + "="*60)
        print("TRAINING: DECISION TREE")
        print("="*60)
        
        # Create model
        model = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced'
        )
        
        # Train
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test,
            "Decision Tree"
        )
        
        # Store
        self.models['decision_tree'] = model
        self.results['decision_tree'] = results
        
        return results
    
    def _evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """
        Evaluate model performance
        
        Returns:
            Dictionary with all metrics
        """
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
        
        # Check for overfitting
        overfit_diff = train_acc - test_acc
        if overfit_diff > 0.05:
            print(f"\n⚠️  Warning: Possible overfitting detected")
            print(f"   Train-Test gap: {overfit_diff:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_test, 
                                   target_names=['No Claim', 'Claim']))
        
        # Return results dictionary
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
        """
        Get the best performing baseline model
        
        Returns:
            Name and results of best model
        """
        if not self.results:
            print("No models trained yet!")
            return None
        
        best_name = max(self.results, key=lambda k: self.results[k]['roc_auc'])
        best_results = self.results[best_name]
        
        print(f"\n🏆 Best Baseline Model: {best_results['model_name']}")
        print(f"   ROC-AUC: {best_results['roc_auc']:.4f}")
        
        return best_name, best_results


if __name__ == "__main__":
    """Test baseline models"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    print("Testing Baseline Models...")
    
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    print(f"Data loaded: Train={X_train.shape}, Test={X_test.shape}")
    
    # Create baseline models
    baseline = BaselineModels(random_state=42)
    
    # Train Logistic Regression
    lr_results = baseline.train_logistic_regression(X_train, y_train, X_test, y_test)
    
    # Train Decision Tree
    dt_results = baseline.train_decision_tree(X_train, y_train, X_test, y_test)
    
    # Get best baseline
    best_name, best_results = baseline.get_best_model()
    
    print("\n✓ Baseline models test successful!")