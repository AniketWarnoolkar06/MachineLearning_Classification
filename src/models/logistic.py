import joblib
import numpy as np
from pathlib import Path
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def train_logistic_with_tuning(
    preprocess_pipeline,
    X_train,
    y_train,
):
    """
    Train Logistic Regression with class weighting and hyperparameter tuning.
    Returns the best fitted pipeline.
    """

    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocess_pipeline),
            ("classifier", log_reg),
        ]
    )

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=1,     # notebook-safe
        verbose=1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def evaluate_with_threshold(model, X_test, y_test, threshold):
    """
    Evaluate a fitted model at a custom probability threshold.
    """

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    return metrics


def find_best_threshold(model, X_test, y_test):
    """
    Find probability threshold that maximizes F1.
    """

    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1_scores.append(f1_score(y_test, preds))

    best_idx = np.argmax(f1_scores)

    return thresholds[best_idx], f1_scores[best_idx]


def save_logistic_model(model, threshold, path_prefix="models/logistic"):
    """
    Save trained model and threshold relative to project root.
    """

    # Project root = folder containing src/
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    save_model_path = PROJECT_ROOT / f"{path_prefix}_model.pkl"
    save_threshold_path = PROJECT_ROOT / f"{path_prefix}_threshold.txt"

    # Create models/ directory if missing
    save_model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, save_model_path)

    with open(save_threshold_path, "w") as f:
        f.write(str(threshold))

