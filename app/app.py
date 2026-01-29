import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
    classification_report,
)

# ---------------------------------------------------------
# Path handling (so src/ imports work on Streamlit Cloud)
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# App Config
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Model Evaluation",
    layout="wide",
)

st.title("📊 Bank Marketing — Model Evaluation Dashboard")

st.markdown(
    """
Upload a **test CSV file** (must contain column `y` as ground truth).
Then select a trained model to evaluate its performance.
"""
)

# ---------------------------------------------------------
# Model registry
# ---------------------------------------------------------

MODEL_OPTIONS = {
    "Logistic Regression": "logistic",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

@st.cache_resource
def load_model_and_threshold(name: str):

    model_path = PROJECT_ROOT / f"models/{name}_model.pkl"
    threshold_path = PROJECT_ROOT / f"models/{name}_threshold.txt"

    model = joblib.load(model_path)

    with open(threshold_path) as f:
        threshold = float(f.read().strip())

    return model, threshold


def evaluate(model, threshold, X, y):

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y, preds),
        "AUC": roc_auc_score(y, probs),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1": f1_score(y, preds),
        "MCC": matthews_corrcoef(y, preds),
    }

    return metrics, preds, probs


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload test CSV file",
    type=["csv"],
)

model_label = st.selectbox(
    "🤖 Select Model",
    list(MODEL_OPTIONS.keys()),
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    if "y" not in df.columns:
        st.error("Uploaded file must contain target column `y`.")
        st.stop()

    # ---------------------------
    # Clean + map target column
    # ---------------------------
    if df["y"].dtype == object:
        y_clean = df["y"].astype(str).str.strip().str.lower()
        y = y_clean.map({"no": 0, "yes": 1})
    else:
        y = df["y"]

    if y.isna().any():
        st.error(
            "❌ Column 'y' contains invalid values. "
            "Expected only 'yes' or 'no'."
        )
        st.write("Unique values found:", df["y"].unique())
        st.stop()

    X = df.drop(columns=["y"])

    model_name = MODEL_OPTIONS[model_label]

    if st.button("🚀 Evaluate Model"):

        with st.spinner("Loading model and running evaluation..."):

            model, threshold = load_model_and_threshold(model_name)

            metrics, preds, probs = evaluate(
                model,
                threshold,
                X,
                y,
            )

        st.success("Evaluation complete!")

        # ---------------------------
        # Metrics
        # ---------------------------

        st.subheader("📈 Evaluation Metrics")

        cols = st.columns(6)

        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k, f"{v:.4f}")

        # ---------------------------
        # Confusion Matrix
        # ---------------------------

        st.subheader("🧮 Confusion Matrix")

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y,
            preds,
            ax=ax,
            display_labels=["No Subscription", "Subscription"],
        )
        st.pyplot(fig)

        # ---------------------------
        # Classification Report (
        # ---------------------------

        st.subheader("📋 Classification Report")

        report_dict = classification_report(
            y,
            preds,
            output_dict=True,
            target_names=["No Subscription", "Subscription"],
        )

        report_df = pd.DataFrame(report_dict).transpose()

        # Remove the confusing accuracy row
        if "accuracy" in report_df.index:
            report_df = report_df.drop(index="accuracy")

        # Round nicely
        report_df = report_df.round(3)

        st.dataframe(report_df)
