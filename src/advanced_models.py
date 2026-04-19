"""
Advanced Models for Car Insurance Claim Prediction
===================================================
Random Forest, Gradient Boosting, XGBoost, and LightGBM
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ✅ Correct import (outside class)
from imblearn.over_sampling import SMOTE


class AdvancedModels:
    """Train and evaluate advanced ensemble models"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    # ============================================================
    # RANDOM FOREST (SMOTE)
    # ============================================================
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*60)
        print("TRAINING: RANDOM FOREST")
        print("="*60)

        smote = SMOTE(random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(X_res, y_res)

        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test,
            "Random Forest (SMOTE)"
        )

        self.models['random_forest'] = model
        self.results['random_forest'] = results

        return results

    # ============================================================
    # GRADIENT BOOSTING (SMOTE)
    # ============================================================
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*60)
        print("TRAINING: GRADIENT BOOSTING")
        print("="*60)

        smote = SMOTE(random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )

        model.fit(X_res, y_res)

        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test,
            "Gradient Boosting (SMOTE)"
        )

        self.models['gradient_boosting'] = model
        self.results['gradient_boosting'] = results

        return results

    # ============================================================
    # XGBOOST (class weight + optional SMOTE)
    # ============================================================
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*60)
        print("TRAINING: XGBOOST")
        print("="*60)

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

        model.fit(X_train, y_train)

        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test,
            "XGBoost (Weighted)"
        )

        self.models['xgboost'] = model
        self.results['xgboost'] = results

        return results

    # ============================================================
    # LIGHTGBM (class weight)
    # ============================================================
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
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

        model.fit(X_train, y_train)

        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test,
            "LightGBM (Weighted)"
        )

        self.models['lightgbm'] = model
        self.results['lightgbm'] = results

        return results

    # ============================================================
    # EVALUATION
    # ============================================================
    def _evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name):

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred_test)

        print(f"\n{'─'*60}")
        print(f"RESULTS: {model_name}")
        print(f"{'─'*60}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"F1-Score:       {f1:.4f}")
        print(f"ROC-AUC:        {roc_auc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))

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

    # ============================================================
    # BEST MODEL (based on F1, not ROC-AUC)
    # ============================================================
    def get_best_model(self):

        if not self.results:
            print("No models trained yet!")
            return None

        best_name = max(self.results, key=lambda k: self.results[k]['f1_score'])
        best_results = self.results[best_name]

        print(f"\n🏆 Best Advanced Model: {best_results['model_name']}")
        print(f"   F1-Score: {best_results['f1_score']:.4f}")

        return best_name, best_results