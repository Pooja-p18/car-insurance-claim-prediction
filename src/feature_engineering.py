"""
Feature Engineering & Selection
================================
Advanced feature engineering and selection techniques
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureEngineer:
    """Feature engineering and selection"""
    
    def __init__(self):
        self.selected_features = None
        self.feature_scores = None
        
    def create_interaction_features(self, df):
        """
        Create interaction features from important variables
        
        Examples:
        - age_of_car × age_of_policyholder
        - airbags × ncap_rating (safety score)
        - max_power × displacement (power ratio)
        """
        print("\n" + "="*60)
        print("CREATING INTERACTION FEATURES")
        print("="*60)
        
        df_new = df.copy()
        
        # Age interactions
        if 'age_of_car' in df.columns and 'age_of_policyholder' in df.columns:
            df_new['age_interaction'] = df['age_of_car'] * df['age_of_policyholder']
            print("✓ Created: age_interaction (car age × policyholder age)")
        
        # Safety score
        if 'airbags' in df.columns and 'ncap_rating' in df.columns:
            df_new['safety_score'] = df['airbags'] + df['ncap_rating']
            print("✓ Created: safety_score (airbags + ncap_rating)")
        
        # Power to displacement ratio
        if 'max_power' in df.columns and 'displacement' in df.columns:
            df_new['power_per_cc'] = df['max_power'] / (df['displacement'] + 1)
            print("✓ Created: power_per_cc (power/displacement)")
        
        # Torque to power ratio
        if 'max_torque' in df.columns and 'max_power' in df.columns:
            df_new['torque_to_power'] = df['max_torque'] / (df['max_power'] + 1)
            print("✓ Created: torque_to_power (torque/power)")
        
        # Vehicle dimensions
        if all(col in df.columns for col in ['length', 'width', 'height']):
            df_new['vehicle_volume'] = df['length'] * df['width'] * df['height']
            print("✓ Created: vehicle_volume (length × width × height)")
        
        # Policy risk score
        if 'policy_tenure' in df.columns and 'age_of_policyholder' in df.columns:
            df_new['policy_risk'] = df['policy_tenure'] / (df['age_of_policyholder'] + 1)
            print("✓ Created: policy_risk (tenure/age)")
        
        # Population density × area cluster (if numeric)
        if 'population_density' in df.columns and 'area_cluster' in df.columns:
            # Only if area_cluster is numeric
            if pd.api.types.is_numeric_dtype(df['area_cluster']):
                df_new['urban_factor'] = df['population_density'] * df['area_cluster']
                print("✓ Created: urban_factor (density × cluster)")
        
        new_features = len(df_new.columns) - len(df.columns)
        print(f"\n✓ Total new features created: {new_features}")
        
        return df_new
    
    def create_polynomial_features(self, df, features, degree=2):
        """
        Create polynomial features for specific columns
        
        Args:
            df: DataFrame
            features: List of feature names
            degree: Polynomial degree (2 for squares)
        """
        print("\n" + "="*60)
        print(f"CREATING POLYNOMIAL FEATURES (degree={degree})")
        print("="*60)
        
        df_new = df.copy()
        
        for feat in features:
            if feat in df.columns:
                for d in range(2, degree + 1):
                    new_col = f'{feat}_pow{d}'
                    df_new[new_col] = df[feat] ** d
                    print(f"✓ Created: {new_col}")
        
        return df_new
    
    def select_features_statistical(self, X, y, k=20):    # finding top k features using ANOVA & Mutual Info # Y 
        """
        Select top K features using statistical tests
        
        Methods:
        1. F-statistic (ANOVA)
        2. Mutual Information
        """
        print("\n" + "="*60)
        print(f"STATISTICAL FEATURE SELECTION (Top {k})")
        print("="*60)
        
        # Method 1: F-statistic  # Linear relationship between feature & target
        print("\n[1] F-statistic (ANOVA):")
        selector_f = SelectKBest(f_classif, k=k)     # f_classif for classification tasks
        selector_f.fit(X, y)    
        
        f_scores = pd.DataFrame({
            'Feature': X.columns,   
            'F_Score': selector_f.scores_   
        }).sort_values('F_Score', ascending=False)
        
        print(f_scores.head(15).to_string(index=False))
        
        # Method 2: Mutual Information   # Any kind of relationship b/w feature & target including non-linear
        print("\n[2] Mutual Information:")
        selector_mi = SelectKBest(mutual_info_classif, k=k)  
        selector_mi.fit(X, y)
        
        mi_scores = pd.DataFrame({
            'Feature': X.columns,
            'MI_Score': selector_mi.scores_
        }).sort_values('MI_Score', ascending=False)                
        
        print(mi_scores.head(15).to_string(index=False))
        
        # Combine scores
        combined = f_scores.merge(mi_scores, on='Feature')
        combined['Combined_Score'] = (                       # averaging normalized scores
            combined['F_Score'] / combined['F_Score'].max() +
            combined['MI_Score'] / combined['MI_Score'].max()
        ) / 2
        combined = combined.sort_values('Combined_Score', ascending=False)
        
        print("\n[3] Combined Ranking:")
        print(combined.head(k).to_string(index=False))
        
        # Select top K
        self.selected_features = combined.head(k)['Feature'].tolist()
        self.feature_scores = combined
        
        print(f"\n✓ Selected {len(self.selected_features)} features")
        
        return self.selected_features, combined
    
    def select_features_rfe(self, X, y, n_features=20): 
        """
        Recursive Feature Elimination using Random Forest
        """
        print("\n" + "="*60)
        print(f"RECURSIVE FEATURE ELIMINATION (Top {n_features})")
        print("="*60)
        
        # Use Random Forest as estimator
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # RFE initialization
        print("Running RFE (this may take a few minutes)...")
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        
        # Get selected features
        rfe_features = pd.DataFrame({
            'Feature': X.columns,
            'Selected': rfe.support_,    # True/False
            'Ranking': rfe.ranking_      # 1 is best and 2,3 is eliminated
        }).sort_values('Ranking')
        
        print("\nRFE Results:")
        print(rfe_features.head(20).to_string(index=False))
        
        selected = rfe_features[rfe_features['Selected']]['Feature'].tolist()
        
        print(f"\n✓ Selected {len(selected)} features using RFE")
        
        return selected, rfe_features
    
    def select_features_importance(self, X, y, threshold=0.01): # keep a strong model
        """
        Select features using Random Forest feature importance
        """
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE BASED SELECTION")
        print("="*60)
        
        # Train Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get importances
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 20 Important Features:")
        print(importances.head(20).to_string(index=False))
        
        # Select features above threshold
        selected = importances[importances['Importance'] >= threshold]['Feature'].tolist()
        
        print(f"\n✓ Selected {len(selected)} features with importance >= {threshold}")
        
        # Visualize
        plt.figure(figsize=(10, 12))
        top_20 = importances.head(20)
        plt.barh(range(len(top_20)), top_20['Importance'])
        plt.yticks(range(len(top_20)), top_20['Feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('outputs/feature_importance_selection.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: outputs/feature_importance_selection.png")
        plt.close()
        
        return selected, importances
    
    def visualize_feature_scores(self, scores_df):
        """Visualize feature selection scores"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # F-Score
        top_f = scores_df.nlargest(15, 'F_Score')
        axes[0].barh(range(len(top_f)), top_f['F_Score'])
        axes[0].set_yticks(range(len(top_f)))
        axes[0].set_yticklabels(top_f['Feature'])
        axes[0].set_xlabel('F-Score')
        axes[0].set_title('Top 15 by F-Statistic', fontweight='bold')
        axes[0].invert_yaxis()
        
        # MI Score
        top_mi = scores_df.nlargest(15, 'MI_Score')
        axes[1].barh(range(len(top_mi)), top_mi['MI_Score'])
        axes[1].set_yticks(range(len(top_mi)))
        axes[1].set_yticklabels(top_mi['Feature'])
        axes[1].set_xlabel('Mutual Information Score')
        axes[1].set_title('Top 15 by Mutual Information', fontweight='bold')
        axes[1].invert_yaxis()
        
        # Combined
        top_combined = scores_df.nlargest(15, 'Combined_Score')
        axes[2].barh(range(len(top_combined)), top_combined['Combined_Score'])
        axes[2].set_yticks(range(len(top_combined)))
        axes[2].set_yticklabels(top_combined['Feature'])
        axes[2].set_xlabel('Combined Score')
        axes[2].set_title('Top 15 by Combined Score', fontweight='bold')
        axes[2].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('outputs/feature_selection_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: outputs/feature_selection_comparison.png")
        plt.close()


if __name__ == "__main__":
    """Test feature engineering"""
    print("Testing Feature Engineering...")
    
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    print(f"Original data: {X_train.shape}")
    
    # Create feature engineer
    fe = FeatureEngineer()
    
    # Create interaction features
    X_engineered = fe.create_interaction_features(X_train)
    print(f"\nAfter feature engineering: {X_engineered.shape}")
    
    # Feature selection - Statistical
    selected_stat, scores = fe.select_features_statistical(X_engineered, y_train, k=30)
    
    # Feature selection - Importance
    selected_imp, importances = fe.select_features_importance(X_engineered, y_train, threshold=0.01)
    
    # Visualize
    fe.visualize_feature_scores(scores)
    
    print("\n✓ Feature engineering test complete!")