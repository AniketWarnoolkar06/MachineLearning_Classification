from pathlib import Path
import numpy as np
import joblib

from sklearn.neighbors import KNeighborsClassifier
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
# Train KNN with CV tuning
# ---------------------------------------------------------

def train_knn(
    preprocess_pipeline,
    X_train,
    y_train,
):
    """
    Train KNN with cross-validated hyperparameter tuning.
    """

    knn = KNeighborsClassifier()

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocess_pipeline),
            ("classifier", knn),
        ]
    )

    param_grid = {
        "classifier__n_neighbors": [5, 10, 25, 50],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=1,
        verbose=1
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

    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1_scores.append(f1_score(y_test, preds))

    best_idx = np.argmax(f1_scores)

    return thresholds[best_idx], f1_scores[best_idx]


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_knn(model, X_test, y_test, threshold):
    """
    Evaluate KNN at custom threshold.
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

def save_knn(model, threshold, path_prefix="models/knn"):
    """
    Save trained KNN model and threshold to project root.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    model_path = PROJECT_ROOT / f"{path_prefix}_model.pkl"
    threshold_path = PROJECT_ROOT / f"{path_prefix}_threshold.txt"

    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    with open(threshold_path, "w") as f:
        f.write(str(threshold))