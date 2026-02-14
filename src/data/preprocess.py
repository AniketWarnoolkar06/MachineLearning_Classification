import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "pdays" in X.columns:
            X["prev_contacted"] = (X["pdays"] != -1).astype(int)

        if "campaign" in X.columns:
            X["campaign_bin"] = pd.cut(
                X["campaign"],
                bins=[0, 1, 3, np.inf],
                labels=["once", "few", "many"]
            ).astype(str)

        if "age" in X.columns:
            X["age_group"] = pd.cut(
                X["age"],
                bins=[0, 30, 45, 60, 100],
                labels=["young", "mid", "senior", "old"]
            ).astype(str)
        
        # Clip extreme values
        for col in ["balance", "campaign", "pdays", "previous"]:
            if col in X.columns:
                X[col] = X[col].clip(lower=X[col].quantile(0.01), upper=X[col].quantile(0.99))

        # Log transform
        if "balance" in X.columns:
            X["balance_log"] = np.sign(X["balance"]) * np.log1p(np.abs(X["balance"]))
            X = X.drop(columns=["balance"])

        return X


def split_data(df):
    """
    Splits data into train/test sets.
    Removes leakage column 'duration'.
    Encodes target variable.
    """
    X = df.drop(columns=["y", "duration"])
    y = df["y"].map({"no": 0, "yes": 1})

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def build_baseline_preprocessor(num_cols, cat_cols):
    num_cols_local = num_cols + ["balance_log"]
    cat_cols_local = cat_cols + ["campaign_bin", "age_group", "prev_contacted"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols_local),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_local),
        ]
    )



def build_enhanced_pipeline(num_cols, cat_cols):
    # along with engineered features
    num_cols_local = num_cols + ["balance_log"]
    cat_cols_local = cat_cols + ["campaign_bin", "age_group", "prev_contacted"]
    return Pipeline(
        steps=[
            ("feature_engineering", BankFeatureEngineer()),
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), num_cols_local),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_local),
                    ]
                ),
            ),
        ]
    )


def build_non_scaled_pipeline(num_cols, cat_cols):
    # Add engineered features
    num_cols_local = num_cols + ["balance_log"]
    cat_cols_local = cat_cols + ["campaign_bin", "age_group", "prev_contacted"]
    return Pipeline(
        steps=[
            ("feature_engineering", BankFeatureEngineer()),
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", "passthrough", num_cols_local),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_local),
                    ]
                ),
            ),
        ]
    )

