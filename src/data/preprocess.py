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

        X["prev_contacted"] = (X["pdays"] != -1).astype(int)

        X["campaign_bin"] = pd.cut(
            X["campaign"],
            bins=[0, 1, 3, np.inf],
            labels=["once", "few", "many"]
        )

        X["age_group"] = pd.cut(
            X["age"],
            bins=[0, 30, 45, 60, 100],
            labels=["young", "mid", "senior", "old"]
        )

        X["balance_log"] = np.sign(X["balance"]) * np.log1p(np.abs(X["balance"]))

        return X


def split_data(df):
    X = df.drop(columns=["y", "duration"])
    y = df["y"].map({"no": 0, "yes": 1})

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def build_baseline_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def build_enhanced_pipeline(num_cols, cat_cols):
    return Pipeline(
        steps=[
            ("feature_engineering", BankFeatureEngineer()),
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), num_cols),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                    ]
                ),
            ),
        ]
    )


def build_tree_nb_pipeline(num_cols, cat_cols):
    """
    Preprocessing pipeline WITHOUT scaling for tree-based models
    and Naive Bayes.
    """

    return Pipeline(
        steps=[
            ("feature_engineering", BankFeatureEngineer()),
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", "passthrough", num_cols),   # NO scaling
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                    ]
                ),
            ),
        ]
    )
