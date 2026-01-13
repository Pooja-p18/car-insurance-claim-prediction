
"""
Main Training Script - Car Insurance Claim Prediction
======================================================
Complete pipeline: Load → Preprocess → Train → Evaluate → Save
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.preprocessing import InsurancePreprocessor
from src.baseline_models import BaselineModels
from src.advanced_models import AdvancedModels

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("CAR INSURANCE CLAIM PREDICTION - TRAINING PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1/6] LOADING DATA")
print("-"*80)

df = pd.read_csv('data/raw/train.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"✓ Target distribution: {df['is_claim'].value_counts().to_dict()}")

# ============================================================================
# STEP 2: PREPROCESS DATA
# ============================================================================
print("\n[STEP 2/6] PREPROCESSING DATA")
print("-"*80)

# Create preprocessor
preprocessor = InsurancePreprocessor()

# Fit and transform
X, y = preprocessor.fit_transform(df, target_col='is_claim')

print(f"✓ Preprocessing complete!")
print(f"✓ Features: {X.shape}")
print(f"✓ Target: {y.shape}")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
print("\n[STEP 3/6] TRAIN-TEST SPLIT")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y  # Maintain class distribution
)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")
print(f"✓ Class distribution maintained: {np.bincount(y_train) / len(y_train)}")

# Save processed data
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
pd.DataFrame(y_train, columns=['is_claim']).to_csv('data/processed/y_train.csv', index=False)
pd.DataFrame(y_test, columns=['is_claim']).to_csv('data/processed/y_test.csv', index=False)
print("✓ Processed data saved to data/processed/")

# Save preprocessor
preprocessor.save('models/preprocessor.pkl')

# ============================================================================
# STEP 4: TRAIN BASELINE MODELS
# ============================================================================
print("\n[STEP 4/6] TRAINING BASELINE MODELS")
print("-"*80)

baseline = BaselineModels(random_state=RANDOM_STATE)

# Logistic Regression
lr_results = baseline.train_logistic_regression(X_train, y_train, X_test, y_test)

# Decision Tree
dt_results = baseline.train_decision_tree(X_train, y_train, X_test, y_test)

# Get best baseline
best_baseline_name, best_baseline_results = baseline.get_best_model()

# ============================================================================
# STEP 5: TRAIN ADVANCED MODELS
# ============================================================================
print("\n[STEP 5/6] TRAINING ADVANCED MODELS")
print("-"*80)

advanced = AdvancedModels(random_state=RANDOM_STATE)

# Random Forest
rf_results = advanced.train_random_forest(X_train, y_train, X_test, y_test)

# Gradient Boosting
gb_results = advanced.train_gradient_boosting(X_train, y_train, X_test, y_test)

# XGBoost
xgb_results = advanced.train_xgboost(X_train, y_train, X_test, y_test)

# LightGBM
lgbm_results = advanced.train_lightgbm(X_train, y_train, X_test, y_test)

# Get best advanced model
best_advanced_name, best_advanced_results = advanced.get_best_model()

# ============================================================================
# STEP 6: COMPARE ALL MODELS & SELECT BEST
# ============================================================================
print("\n[STEP 6/6] MODEL COMPARISON & SELECTION")
print("="*80)

# Combine all results
all_results = []
all_results.extend([lr_results, dt_results])
all_results.extend([rf_results, gb_results, xgb_results, lgbm_results])

# Create comparison dataframe
comparison_df = pd.DataFrame([{
    'Model': r['model_name'],
    'Train Acc': r['train_accuracy'],
    'Test Acc': r['test_accuracy'],
    'Precision': r['precision'],
    'Recall': r['recall'],
    'F1-Score': r['f1_score'],
    'ROC-AUC': r['roc_auc']
} for r in all_results])

comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)

print("\n📊 MODEL PERFORMANCE COMPARISON")
print(comparison_df.to_string(index=False))

# Select best model overall
best_model_idx = comparison_df['ROC-AUC'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_roc_auc = comparison_df.loc[best_model_idx, 'ROC-AUC']

# Find the actual model object
best_model = None
for result in all_results:
    if result['model_name'] == best_model_name:
        best_model = result['model']
        break

print(f"\n{'='*80}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"{'='*80}")
print(f"Test Accuracy:  {comparison_df.loc[best_model_idx, 'Test Acc']:.4f}")
print(f"Precision:      {comparison_df.loc[best_model_idx, 'Precision']:.4f}")
print(f"Recall:         {comparison_df.loc[best_model_idx, 'Recall']:.4f}")
print(f"F1-Score:       {comparison_df.loc[best_model_idx, 'F1-Score']:.4f}")
print(f"ROC-AUC:        {best_roc_auc:.4f}")

# ============================================================================
# SAVE BEST MODEL
# ============================================================================
print("\n" + "="*80)
print("SAVING MODEL & RESULTS")
print("="*80)

# Save best model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Best model saved: models/best_model.pkl")

# Save comparison results
comparison_df.to_csv('outputs/model_comparison.csv', index=False)
print("✓ Comparison saved: outputs/model_comparison.csv")

# Save feature names
with open('models/feature_names.txt', 'w') as f:
    f.write('\n'.join(preprocessor.feature_columns))
print("✓ Feature names saved: models/feature_names.txt")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Model Comparison Plot
fig, ax = plt.subplots(figsize=(12, 6))
metrics_df = comparison_df.set_index('Model')[['Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
metrics_df.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim([0, 1])
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/model_comparison.png")

# 2. Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': preprocessor.feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot top 20
    plt.figure(figsize=(10, 12))
    top_20 = feature_imp_df.head(20)
    plt.barh(range(len(top_20)), top_20['Importance'])
    plt.yticks(range(len(top_20)), top_20['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 20 Feature Importances - {best_model_name}', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/feature_importance.png")
    
    # Save to CSV
    feature_imp_df.to_csv('outputs/feature_importance.csv', index=False)
    print("✓ Saved: outputs/feature_importance.csv")

plt.close('all')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE! 🎉")
print("="*80)

print(f"""
📊 SUMMARY:
  • Dataset: {len(df)} records
  • Features: {X.shape[1]} 
  • Models Trained: {len(all_results)}
  
🏆 BEST MODEL: {best_model_name}
  • Test Accuracy: {comparison_df.loc[best_model_idx, 'Test Acc']:.2%}
  • ROC-AUC: {best_roc_auc:.4f}
  • F1-Score: {comparison_df.loc[best_model_idx, 'F1-Score']:.4f}

📁 SAVED FILES:
  ✓ models/preprocessor.pkl (preprocessing pipeline)
  ✓ models/best_model.pkl ({best_model_name})
  ✓ models/feature_names.txt
  ✓ outputs/model_comparison.csv
  ✓ outputs/model_comparison.png
  ✓ outputs/feature_importance.png
  ✓ data/processed/* (train/test splits)

🚀 NEXT STEPS:
  1. Review model comparison results
  2. Analyze feature importance
  3. Test Streamlit app: streamlit run streamlit_app.py
  4. Deploy to cloud (Streamlit Cloud/Heroku)

✅ Ready for deployment!
""")

print("="*80)