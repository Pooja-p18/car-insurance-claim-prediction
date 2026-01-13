"""
Quick EDA Script - Car Insurance Claim Prediction
==================================================
Run this to generate all EDA visualizations quickly

Usage: python eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Create outputs folder if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("RUNNING EDA - CAR INSURANCE CLAIM PREDICTION")
print("="*80)

# Load data
print("\n[1/8] Loading data...")
df = pd.read_csv('data/raw/train.csv')
print(f"✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Target distribution
print("\n[2/8] Analyzing target variable...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
claim_counts = df['is_claim'].value_counts()
claim_pct = df['is_claim'].value_counts(normalize=True) * 100

axes[0].bar(['No Claim', 'Claim'], claim_counts, color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[0].set_title('Claim Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(claim_counts):
    axes[0].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')

axes[1].pie(claim_counts, labels=['No Claim', 'Claim'], autopct='%1.2f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0.05, 0.05))
axes[1].set_title('Claim Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_target_distribution.png")
print(f"  Claim Rate: {claim_pct[1]:.2f}%")

# Missing values
print("\n[3/8] Analyzing missing values...")
missing = df.isnull().sum()
missing_pct = 100 * missing / len(df)
missing_df = pd.DataFrame({'Column': missing.index, 'Missing_%': missing_pct.values})
missing_df = missing_df[missing_df['Missing_%'] > 0].sort_values('Missing_%', ascending=False)

if len(missing_df) > 0:
    plt.figure(figsize=(12, 6))
    plt.barh(missing_df['Column'], missing_df['Missing_%'], color='coral')
    plt.xlabel('Missing Percentage (%)')
    plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('outputs/02_missing_values.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 02_missing_values.png")
    print(f"  Found missing values in {len(missing_df)} columns")
else:
    print("✓ No missing values found")

# Numerical features
print("\n[4/8] Analyzing numerical features...")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [c for c in numerical_cols if c not in ['policy_id', 'is_claim']]

n_cols = 4
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    if idx < len(axes):
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2)
        axes[idx].axvline(df[col].median(), color='green', linestyle='--', linewidth=2)
        axes[idx].set_title(col, fontsize=10, fontweight='bold')

for idx in range(len(numerical_cols), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Numerical Features Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/03_numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: 03_numerical_distributions.png")
print(f"  Analyzed {len(numerical_cols)} numerical features")

# Categorical features
print("\n[5/8] Analyzing categorical features...")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

n_cols = 3
n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows*4))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols):
    if idx < len(axes):
        top_10 = df[col].value_counts().head(10)
        top_10.plot(kind='barh', ax=axes[idx], color='coral', edgecolor='black')
        axes[idx].set_title(f'{col}\n({df[col].nunique()} categories)', fontsize=10, fontweight='bold')
        axes[idx].invert_yaxis()

for idx in range(len(categorical_cols), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Categorical Features Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/04_categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: 04_categorical_distributions.png")
print(f"  Analyzed {len(categorical_cols)} categorical features")

# Correlation analysis
print("\n[6/8] Analyzing correlations...")
correlations = df[numerical_cols + ['is_claim']].corr()['is_claim'].abs().sort_values(ascending=False)
top_features = correlations.head(16).index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Heatmap
corr_matrix = df[top_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, ax=axes[0])
axes[0].set_title('Correlation Heatmap - Top 15 Features', fontsize=14, fontweight='bold')

# Bar plot
target_corr = correlations.drop('is_claim').head(15)
axes[1].barh(range(len(target_corr)), target_corr.values, color='steelblue')
axes[1].set_yticks(range(len(target_corr)))
axes[1].set_yticklabels(target_corr.index)
axes[1].set_xlabel('Correlation with Target')
axes[1].set_title('Top 15 Features by Correlation', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/05_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_correlation_analysis.png")
print(f"  Top correlated feature: {target_corr.index[0]} ({target_corr.values[0]:.4f})")

# Bivariate - Numerical
print("\n[7/8] Bivariate analysis (Numerical)...")
top_6 = correlations.drop('is_claim').head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, col in enumerate(top_6):
    df.boxplot(column=col, by='is_claim', ax=axes[idx])
    axes[idx].set_title(f'{col} by Claim Status')
    plt.sca(axes[idx])
    plt.xticks([1, 2], ['No Claim', 'Claim'])

plt.suptitle('Top Features by Claim Status', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/06_bivariate_numerical.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_bivariate_numerical.png")

# Bivariate - Categorical
print("\n[8/8] Bivariate analysis (Categorical)...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols[:6]):
    claim_rate = df.groupby(col)['is_claim'].mean().sort_values(ascending=False).head(10)
    claim_rate.plot(kind='barh', ax=axes[idx], color='coral', edgecolor='black')
    axes[idx].set_title(f'Claim Rate by {col}', fontweight='bold')
    axes[idx].axvline(df['is_claim'].mean(), color='red', linestyle='--', linewidth=2)

plt.suptitle('Claim Rate by Categorical Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/07_bivariate_categorical.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_bivariate_categorical.png")

# Summary
print("\n" + "="*80)
print("EDA COMPLETE! ✅")
print("="*80)
print(f"""
SUMMARY:
• Dataset: {len(df):,} rows × {len(df.columns)} columns
• Claim Rate: {claim_pct[1]:.2f}%
• Numerical Features: {len(numerical_cols)}
• Categorical Features: {len(categorical_cols)}
• Top Predictor: {target_corr.index[0]} (correlation: {target_corr.values[0]:.4f})

VISUALIZATIONS SAVED:
""")

output_files = sorted([f for f in os.listdir('outputs') if f.endswith('.png')])
for i, f in enumerate(output_files, 1):
    print(f"  {i}. outputs/{f}")

print(f"""
NEXT STEPS:
1. Review all visualizations in 'outputs/' folder
2. Run preprocessing and training: python train.py
3. Launch Streamlit app: streamlit run streamlit_app.py

{"="*80}
""")