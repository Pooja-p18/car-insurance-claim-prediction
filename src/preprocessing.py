"""
Preprocessor Class for Car Insurance Claim Prediction
======================================================
Production-ready preprocessing pipeline
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


class InsurancePreprocessor:
    """
    Complete preprocessing pipeline that handles:
    - Missing values
    - Categorical encoding
    - Feature scaling
    - Column consistency between training & prediction
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.numerical_features = []
        self.categorical_features = []
        self.fill_values = {}

    # --------------------------------------------------
    # FEATURE IDENTIFICATION
    # --------------------------------------------------
    def identify_features(self, df):
        exclude_cols = ['is_claim', 'policy_id']

        self.numerical_features = [
            c for c in df.select_dtypes(include=['int64', 'float64']).columns
            if c not in exclude_cols
        ]

        self.categorical_features = [
            c for c in df.select_dtypes(include=['object', 'category']).columns
            if c not in exclude_cols
        ]

        print(f"✓ Numerical features: {len(self.numerical_features)}")
        print(f"✓ Categorical features: {len(self.categorical_features)}")

    # --------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------
    def handle_missing(self, df, is_training=True):
        df = df.copy()

        if is_training:
            for col in self.numerical_features:
                if col in df:
                    self.fill_values[col] = df[col].median()

            for col in self.categorical_features:
                if col in df:
                    mode = df[col].mode()
                    self.fill_values[col] = mode[0] if len(mode) else "Unknown"

        for col in self.numerical_features:
            if col in df:
                df[col] = df[col].fillna(self.fill_values.get(col, 0))

        for col in self.categorical_features:
            if col in df:
                df[col] = df[col].fillna(self.fill_values.get(col, "Unknown"))

        return df

    # --------------------------------------------------
    # CATEGORICAL ENCODING
    # --------------------------------------------------
    def encode_categorical(self, df, is_training=True):
        df = df.copy()

        for col in self.categorical_features:
            if col not in df:
                continue

            df[col] = df[col].astype(str)

            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(f"  ✓ Encoded {col}: {len(le.classes_)} categories")
            else:
                le = self.label_encoders[col]
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        return df

    # --------------------------------------------------
    # SCALING
    # --------------------------------------------------
    def scale_features(self, df, is_training=True):
        df = df.copy()

        if is_training:
            df[self.numerical_features] = self.scaler.fit_transform(
                df[self.numerical_features]
            )
            print(f"  ✓ Scaled {len(self.numerical_features)} numerical features")
        else:
            df[self.numerical_features] = self.scaler.transform(
                df[self.numerical_features]
            )

        return df

    # --------------------------------------------------
    # TRAINING PIPELINE
    # --------------------------------------------------
    def fit_transform(self, df, target_col="is_claim"):
        print("\n" + "=" * 60)
        print("FITTING PREPROCESSOR (TRAINING MODE)")
        print("=" * 60)

        df = df.copy()
        if "policy_id" in df:
            df = df.drop(columns=["policy_id"])

        self.identify_features(df)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = self.handle_missing(X, is_training=True)
        X = self.encode_categorical(X, is_training=True)
        X = self.scale_features(X, is_training=True)

        self.feature_columns = X.columns.tolist()

        print("\n✓ Pipeline fitted successfully!")
        print(f"✓ Total features: {len(self.feature_columns)}")

        return X, y

    # --------------------------------------------------
    # PREDICTION PIPELINE (FIXED)
    # --------------------------------------------------
    def transform(self, df):
        print("\n" + "=" * 60)
        print("TRANSFORMING DATA (PREDICTION MODE)")
        print("=" * 60)

        df = df.copy()

        # Remove unwanted columns
        df = df.drop(columns=[c for c in ["policy_id", "is_claim"] if c in df], errors="ignore")

        # 🔑 ADD MISSING COLUMNS FIRST
        for col in self.feature_columns:
            if col not in df:
                df[col] = self.fill_values.get(col, 0)

        # Remove extras
        df = df[self.feature_columns]

        # Apply transformations
        df = self.handle_missing(df, is_training=False)
        df = self.encode_categorical(df, is_training=False)
        df = self.scale_features(df, is_training=False)

        print(f"\n✓ Data ready for prediction!")
        print(f"✓ Final shape: {df.shape}")

        return df

    # --------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------
    def save(self, filepath="models/preprocessor.pkl"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)

        print(f"\n✓ Preprocessor saved to: {filepath}")

    @classmethod
    def load(cls, filepath="models/preprocessor.pkl"):
        preprocessor = cls()

        with open(filepath, "rb") as f:
            preprocessor.__dict__ = pickle.load(f)

        print(f"✓ Preprocessor loaded from: {filepath}")
        return preprocessor


# --------------------------------------------------
# TESTING
# --------------------------------------------------
if __name__ == "__main__":
    print("Testing Preprocessor...")

    df = pd.read_csv("data/raw/train.csv")
    print(f"Data loaded: {df.shape}")

    preprocessor = InsurancePreprocessor()
    X, y = preprocessor.fit_transform(df)

    print(f"\nProcessed data: X={X.shape}, y={y.shape}")

    preprocessor.save()

    loaded = InsurancePreprocessor.load()

    sample = pd.DataFrame({
        "age_of_car": [3],
        "segment": ["B1"],
        "fuel_type": ["Petrol"]
    })

    print("\nTesting with sample data:")
    print(sample)

    transformed = loaded.transform(sample)
    print(f"\nTransformed shape: {transformed.shape}")
    print("✓ Preprocessor test successful!")
