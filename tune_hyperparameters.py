"""
Comprehensive Hyperparameter Tuning
====================================
GridSearchCV and RandomizedSearchCV for best model optimization
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HYPERPARAMETER TUNING")
print("="*80)

# Load data
print("\n[1/4] Loading data...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")

# Define scoring
scorer = make_scorer(roc_auc_score, needs_proba=True)

# ============================================================================
# RANDOM FOREST TUNING
# ============================================================================
print("\n[2/4] TUNING RANDOM FOREST")
print("="*80)

# Define parameter grid
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, 25],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}

print(f"Parameter grid size: {np.prod([len(v) for v in rf_param_grid.values()])} combinations")

# Create model
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# GridSearchCV
print("\nRunning GridSearchCV (this will take 10-15 minutes)...")
rf_grid = GridSearchCV(
    rf,
    rf_param_grid,
    cv=3,
    scoring=scorer,
    n_jobs=-1,
    verbose=2
)

rf_grid.fit(X_train, y_train)

print(f"\n✓ Best Random Forest parameters:")
print(rf_grid.best_params_)
print(f"✓ Best CV ROC-AUC: {rf_grid.best_score_:.4f}")

# Test performance
y_pred_proba = rf_grid.best_estimator_.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"✓ Test ROC-AUC: {test_roc_auc:.4f}")

# Save results
rf_results = pd.DataFrame(rf_grid.cv_results_)
rf_results.to_csv('outputs/rf_tuning_results.csv', index=False)

# Save best model
with open('models/rf_tuned.pkl', 'wb') as f:
    pickle.dump(rf_grid.best_estimator_, f)
print("✓ Saved: models/rf_tuned.pkl")

# ============================================================================
# XGBOOST TUNING
# ============================================================================
print("\n[3/4] TUNING XGBOOST")
print("="*80)

# Calculate scale_pos_weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Define parameter grid
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'scale_pos_weight': [scale_pos_weight]
}

print(f"Parameter grid size: {np.prod([len(v) for v in xgb_param_grid.values()])} combinations")

# Use RandomizedSearchCV for faster tuning
print("\nRunning RandomizedSearchCV (n_iter=50)...")
xgb = XGBClassifier(random_state=42, eval_metric='logloss')

xgb_random = RandomizedSearchCV(
    xgb,
    xgb_param_grid,
    n_iter=50,  # Try 50 random combinations
    cv=3,
    scoring=scorer,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

xgb_random.fit(X_train, y_train)

print(f"\n✓ Best XGBoost parameters:")
print(xgb_random.best_params_)
print(f"✓ Best CV ROC-AUC: {xgb_random.best_score_:.4f}")

# Test performance
y_pred_proba = xgb_random.best_estimator_.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"✓ Test ROC-AUC: {test_roc_auc:.4f}")

# Save results
xgb_results = pd.DataFrame(xgb_random.cv_results_)
xgb_results.to_csv('outputs/xgb_tuning_results.csv', index=False)

# Save best model
with open('models/xgb_tuned.pkl', 'wb') as f:
    pickle.dump(xgb_random.best_estimator_, f)
print("✓ Saved: models/xgb_tuned.pkl")

# ============================================================================
# LIGHTGBM TUNING
# ============================================================================
print("\n[4/4] TUNING LIGHTGBM")
print("="*80)

# Define parameter grid
lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'class_weight': ['balanced']
}

print(f"Parameter grid size: {np.prod([len(v) for v in lgbm_param_grid.values()])} combinations")

# Use RandomizedSearchCV
print("\nRunning RandomizedSearchCV (n_iter=50)...")
lgbm = LGBMClassifier(random_state=42, verbose=-1)

lgbm_random = RandomizedSearchCV(
    lgbm,
    lgbm_param_grid,
    n_iter=50,
    cv=3,
    scoring=scorer,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

lgbm_random.fit(X_train, y_train)

print(f"\n✓ Best LightGBM parameters:")
print(lgbm_random.best_params_)
print(f"✓ Best CV ROC-AUC: {lgbm_random.best_score_:.4f}")

# Test performance
y_pred_proba = lgbm_random.best_estimator_.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"✓ Test ROC-AUC: {test_roc_auc:.4f}")

# Save results
lgbm_results = pd.DataFrame(lgbm_random.cv_results_)
lgbm_results.to_csv('outputs/lgbm_tuning_results.csv', index=False)

# Save best model
with open('models/lgbm_tuned.pkl', 'wb') as f:
    pickle.dump(lgbm_random.best_estimator_, f)
print("✓ Saved: models/lgbm_tuned.pkl")

# ============================================================================
# COMPARE ALL TUNED MODELS
# ============================================================================
print("\n" + "="*80)
print("COMPARING TUNED MODELS")
print("="*80)

# Load all tuned models
with open('models/rf_tuned.pkl', 'rb') as f:
    rf_tuned = pickle.load(f)
with open('models/xgb_tuned.pkl', 'rb') as f:
    xgb_tuned = pickle.load(f)
with open('models/lgbm_tuned.pkl', 'rb') as f:
    lgbm_tuned = pickle.load(f)

# Evaluate all
models = {
    'Random Forest (Tuned)': rf_tuned,
    'XGBoost (Tuned)': xgb_tuned,
    'LightGBM (Tuned)': lgbm_tuned
}

results = []
for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    results.append({'Model': name, 'Test_ROC_AUC': roc_auc})

results_df = pd.DataFrame(results).sort_values('Test_ROC_AUC', ascending=False)

print("\nTuned Model Performance:")
print(results_df.to_string(index=False))

# Select best
best_model_name = results_df.iloc[0]['Model']
best_roc_auc = results_df.iloc[0]['Test_ROC_AUC']

print(f"\n🏆 BEST TUNED MODEL: {best_model_name}")
print(f"   Test ROC-AUC: {best_roc_auc:.4f}")

# Save best tuned model as final model
best_model_key = best_model_name.split()[0].lower()
if 'random' in best_model_name.lower():
    best_model = rf_tuned
elif 'xgb' in best_model_name.lower():
    best_model = xgb_tuned
else:
    best_model = lgbm_tuned

with open('models/best_model_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Saved: models/best_model_tuned.pkl")

# ============================================================================
# VISUALIZE TUNING RESULTS
# ============================================================================
print("\nGenerating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Model comparison
axes[0].barh(results_df['Model'], results_df['Test_ROC_AUC'], color='steelblue')
axes[0].set_xlabel('ROC-AUC Score')
axes[0].set_title('Tuned Model Performance Comparison', fontweight='bold')
axes[0].set_xlim([0.7, 1.0])
axes[0].grid(axis='x', alpha=0.3)

# Plot 2: Before vs After tuning
# Load original best model
with open('models/best_model.pkl', 'rb') as f:
    original_model = pickle.load(f)

original_roc = roc_auc_score(y_test, original_model.predict_proba(X_test)[:, 1])

comparison_data = pd.DataFrame({
    'Model': ['Before Tuning', 'After Tuning'],
    'ROC_AUC': [original_roc, best_roc_auc]
})

axes[1].bar(comparison_data['Model'], comparison_data['ROC_AUC'], 
           color=['coral', 'green'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('ROC-AUC Score')
axes[1].set_title('Improvement from Hyperparameter Tuning', fontweight='bold')
axes[1].set_ylim([0.7, 1.0])
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for i, (model, score) in enumerate(zip(comparison_data['Model'], comparison_data['ROC_AUC'])):
    axes[1].text(i, score + 0.01, f'{score:.4f}', ha='center', fontweight='bold')

# Add improvement text
improvement = ((best_roc_auc - original_roc) / original_roc) * 100
axes[1].text(0.5, 0.95, f'Improvement: +{improvement:.2f}%', 
            transform=axes[1].transAxes, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('outputs/hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/hyperparameter_tuning_results.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("HYPERPARAMETER TUNING COMPLETE! ✅")
print("="*80)

print(f"""
RESULTS:
✓ Random Forest - Best CV ROC-AUC: {rf_grid.best_score_:.4f}
✓ XGBoost       - Best CV ROC-AUC: {xgb_random.best_score_:.4f}
✓ LightGBM      - Best CV ROC-AUC: {lgbm_random.best_score_:.4f}

BEST MODEL: {best_model_name}
Test ROC-AUC: {best_roc_auc:.4f}

IMPROVEMENT:
Before Tuning: {original_roc:.4f}
After Tuning:  {best_roc_auc:.4f}
Gain: +{improvement:.2f}%

SAVED FILES:
✓ models/rf_tuned.pkl
✓ models/xgb_tuned.pkl
✓ models/lgbm_tuned.pkl
✓ models/best_model_tuned.pkl (use this for deployment!)
✓ outputs/*_tuning_results.csv

NEXT STEP:
Update streamlit_app.py to use models/best_model_tuned.pkl
""")

print("="*80)