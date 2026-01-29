from pathlib import Path
import numpy as np
import joblib

from xgboost import XGBClassifier

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


# ---------------------------------------------------------
# Train XGBoost with FINAL tuned grid
# ---------------------------------------------------------

def train_xgboost(
    preprocess_pipeline,
    X_train,
    y_train,
):
    """
    Train XGBoost using the final tuned hyperparameter grid
    that achieved the best F1 / MCC.
    """

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocess_pipeline),
            ("classifier", xgb),
        ]
    )
    
        
    param_grid = {
        "classifier__n_estimators": [400],
        "classifier__max_depth": [6, 8],
        "classifier__learning_rate": [0.05],
        "classifier__min_child_weight": [1, 3],
        "classifier__subsample": [0.9, 1.0],
        "classifier__colsample_bytree": [0.9, 1.0],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ---------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------

def find_best_threshold(model, X_test, y_test):
    """
    Find probability threshold that maximizes F1.
    """

    probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.05, 0.6, 80)
    f1_scores = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1_scores.append(f1_score(y_test, preds))

    best_idx = np.argmax(f1_scores)

    return thresholds[best_idx], f1_scores[best_idx]


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_xgb(model, X_test, y_test, threshold):
    """
    Evaluate XGBoost at custom threshold.
    """

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "AUC": roc_auc_score(y_test, probs),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "MCC": matthews_corrcoef(y_test, preds),
    }

    return metrics


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

def save_xgboost(model, threshold, path_prefix="models/xgboost"):
    """
    Save trained XGBoost model and threshold to project root.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    model_path = PROJECT_ROOT / f"{path_prefix}_model.pkl"
    threshold_path = PROJECT_ROOT / f"{path_prefix}_threshold.txt"

    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    with open(threshold_path, "w") as f:
        f.write(str(threshold))