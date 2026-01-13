"""
Comprehensive Model Evaluation
===============================
Generate detailed evaluation report with all metrics
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*80)

# Load data
print("\n[1/6] Loading data and model...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Load best model (use tuned if available)
try:
    with open('models/best_model_tuned.pkl', 'rb') as f:
        model = pickle.load(f)
    model_name = "Best Model (Tuned)"
    print("✓ Loaded: models/best_model_tuned.pkl")
except:
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    model_name = "Best Model"
    print("✓ Loaded: models/best_model.pkl")

# ============================================================================
# PREDICTIONS
# ============================================================================
print("\n[2/6] Generating predictions...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_proba_train = model.predict_proba(X_train)[:, 1]
y_pred_proba_test = model.predict_proba(X_test)[:, 1]
print("✓ Predictions generated")

# ============================================================================
# METRICS CALCULATION
# ============================================================================
print("\n[3/6] Calculating metrics...")

# Train metrics
train_acc = accuracy_score(y_train, y_pred_train)
train_precision = precision_score(y_train, y_pred_train)
train_recall = recall_score(y_train, y_pred_train)
train_f1 = f1_score(y_train, y_pred_train)
train_roc_auc = roc_auc_score(y_train, y_pred_proba_train)

# Test metrics
test_acc = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)
test_avg_precision = average_precision_score(y_test, y_pred_proba_test)

# Confusion matrices
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

print("\n" + "="*80)
print("EVALUATION METRICS")
print("="*80)

print(f"\n{'Metric':<20} {'Train':<12} {'Test':<12} {'Difference':<12}")
print("-"*60)
print(f"{'Accuracy':<20} {train_acc:>11.4f} {test_acc:>11.4f} {train_acc-test_acc:>11.4f}")
print(f"{'Precision':<20} {train_precision:>11.4f} {test_precision:>11.4f} {train_precision-test_precision:>11.4f}")
print(f"{'Recall':<20} {train_recall:>11.4f} {test_recall:>11.4f} {train_recall-test_recall:>11.4f}")
print(f"{'F1-Score':<20} {train_f1:>11.4f} {test_f1:>11.4f} {train_f1-test_f1:>11.4f}")
print(f"{'ROC-AUC':<20} {train_roc_auc:>11.4f} {test_roc_auc:>11.4f} {train_roc_auc-test_roc_auc:>11.4f}")

# Overfitting check
overfit_gap = train_acc - test_acc
if overfit_gap > 0.05:
    print(f"\n⚠️  WARNING: Possible overfitting detected (gap: {overfit_gap:.4f})")
else:
    print(f"\n✓ Good generalization (gap: {overfit_gap:.4f})")

# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================
print("\n[4/6] Creating confusion matrix visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Train confusion matrix
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Claim', 'Claim'],
            yticklabels=['No Claim', 'Claim'])
axes[0].set_title(f'Confusion Matrix - Train\nAccuracy: {train_acc:.4f}', 
                 fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Test confusion matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['No Claim', 'Claim'],
            yticklabels=['No Claim', 'Claim'])
axes[1].set_title(f'Confusion Matrix - Test\nAccuracy: {test_acc:.4f}',
                 fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('outputs/confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/confusion_matrix_detailed.png")
plt.close()

# ============================================================================
# ROC CURVE
# ============================================================================
print("\n[5/6] Creating ROC curve...")

fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)

plt.figure(figsize=(10, 8))
plt.plot(fpr_train, tpr_train, label=f'Train (AUC = {train_roc_auc:.4f})', 
         linewidth=2, color='blue')
plt.plot(fpr_test, tpr_test, label=f'Test (AUC = {test_roc_auc:.4f})',
         linewidth=2, color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/roc_curve_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/roc_curve_detailed.png")
plt.close()

# ============================================================================
# PRECISION-RECALL CURVE
# ============================================================================
print("\n[6/6] Creating precision-recall curve...")

precision_train, recall_train, _ = precision_recall_curve(y_train, y_pred_proba_train)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_proba_test)

plt.figure(figsize=(10, 8))
plt.plot(recall_train, precision_train, 
         label=f'Train (AP = {average_precision_score(y_train, y_pred_proba_train):.4f})',
         linewidth=2, color='blue')
plt.plot(recall_test, precision_test,
         label=f'Test (AP = {test_avg_precision:.4f})',
         linewidth=2, color='green')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/precision_recall_curve.png")
plt.close()

# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT")
print("="*80)

print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred_test, 
                           target_names=['No Claim (0)', 'Claim (1)'],
                           digits=4))

# ============================================================================
# ERROR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("ERROR ANALYSIS")
print("="*80)

# Calculate error types
tn, fp, fn, tp = cm_test.ravel()

print(f"\nConfusion Matrix Breakdown (Test Set):")
print(f"  True Negatives (TN):  {tn:>6,} - Correctly predicted No Claim")
print(f"  False Positives (FP): {fp:>6,} - Predicted Claim, but was No Claim")
print(f"  False Negatives (FN): {fn:>6,} - Predicted No Claim, but was Claim")
print(f"  True Positives (TP):  {tp:>6,} - Correctly predicted Claim")

print(f"\nError Rates:")
print(f"  False Positive Rate: {fp/(fp+tn)*100:>6.2f}% (Type I Error)")
print(f"  False Negative Rate: {fn/(fn+tp)*100:>6.2f}% (Type II Error)")

print(f"\nBusiness Impact:")
print(f"  • {fp:,} customers incorrectly flagged as high risk")
print(f"    → May result in unfair premium increases")
print(f"  • {fn:,} customers incorrectly flagged as low risk")
print(f"    → May result in unexpected claim losses")

# ============================================================================
# SAVE EVALUATION REPORT
# ============================================================================
print("\n" + "="*80)
print("SAVING EVALUATION REPORT")
print("="*80)

# Create comprehensive report
report = {
    'Model': model_name,
    'Train_Accuracy': train_acc,
    'Test_Accuracy': test_acc,
    'Train_Precision': train_precision,
    'Test_Precision': test_precision,
    'Train_Recall': train_recall,
    'Test_Recall': test_recall,
    'Train_F1': train_f1,
    'Test_F1': test_f1,
    'Train_ROC_AUC': train_roc_auc,
    'Test_ROC_AUC': test_roc_auc,
    'Test_Avg_Precision': test_avg_precision,
    'True_Negatives': tn,
    'False_Positives': fp,
    'False_Negatives': fn,
    'True_Positives': tp,
    'FPR': fp/(fp+tn),
    'FNR': fn/(fn+tp),
    'Overfitting_Gap': overfit_gap
}

report_df = pd.DataFrame([report])
report_df.to_csv('outputs/evaluation_report.csv', index=False)
print("✓ Saved: outputs/evaluation_report.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EVALUATION COMPLETE! ✅")
print("="*80)

print(f"""
MODEL: {model_name}

KEY METRICS (Test Set):
  ✓ ROC-AUC:    {test_roc_auc:.4f}
  ✓ Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)
  ✓ Precision:  {test_precision:.4f}
  ✓ Recall:     {test_recall:.4f}
  ✓ F1-Score:   {test_f1:.4f}

CONFUSION MATRIX:
              Predicted
              No    Yes
  Actual No   {tn:>5,} {fp:>5,}
         Yes  {fn:>5,} {tp:>5,}

GENERATED FILES:
  ✓ outputs/confusion_matrix_detailed.png
  ✓ outputs/roc_curve_detailed.png
  ✓ outputs/precision_recall_curve.png
  ✓ outputs/evaluation_report.csv

INTERPRETATION:
  • Model correctly identifies {test_acc*100:.1f}% of cases
  • {test_precision*100:.1f}% of predicted claims are actual claims
  • {test_recall*100:.1f}% of actual claims are detected
  • ROC-AUC of {test_roc_auc:.3f} indicates {"excellent" if test_roc_auc > 0.9 else "good" if test_roc_auc > 0.8 else "fair"} discrimination

READY FOR DEPLOYMENT! ✅
""")

print("="*80)