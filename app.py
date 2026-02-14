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
    RocCurveDisplay,
)

# Path handling to ensure we can load models and data correctly
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# App Config
st.set_page_config(
    page_title="Bank Marketing Model Evaluation",
    layout="wide",
)

st.title("📊 Bank Marketing — Model Evaluation Dashboard")

st.markdown(
    """
### 🎯 Objective
Predict whether a customer will subscribe to a term deposit.

### 📌 Instructions
1. Download sample test dataset OR upload your own
2. Select a trained model
3. Click **Evaluate Model**
"""
)

# Download sample test file
test_file_path = PROJECT_ROOT / "data/test/bank_test.csv"

if test_file_path.exists():
    with open(test_file_path, "rb") as f:
        st.download_button(
            "📥 Download Sample Test File",
            f,
            file_name="bank_test.csv",
        )

st.markdown("---")

# Model registry
MODEL_OPTIONS = {
    "Logistic Regression": "logistic",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}

# Helper Functions
@st.cache_resource
def load_model_and_threshold(name: str):

    model_path = PROJECT_ROOT / f"models/{name}_model.pkl"
    threshold_path = PROJECT_ROOT / f"models/{name}_threshold.txt"

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    if not threshold_path.exists():
        st.error(f"Threshold file not found: {threshold_path}")
        st.stop()

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
        "Precision": precision_score(y, preds, zero_division=0),
        "Recall": recall_score(y, preds, zero_division=0),
        "F1": f1_score(y, preds, zero_division=0),
        "MCC": matthews_corrcoef(y, preds),
    }

    return metrics, preds, probs

# UI Components
uploaded_file = st.file_uploader(
    "📂 Upload Test CSV File",
    type=["csv"],
)

model_label = st.selectbox(
    "🤖 Select Model",
    list(MODEL_OPTIONS.keys()),
)

# Main Logic
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Validate target column
    if "y" not in df.columns:
        st.error("❌ Uploaded file must contain target column 'y'.")
        st.stop()

    # Clean target column
    if df["y"].dtype == object:
        y_clean = df["y"].astype(str).str.strip().str.lower()
        y = y_clean.map({"no": 0, "yes": 1})
    else:
        y = df["y"]

    if y.isna().any():
        st.error(
            "❌ Column 'y' contains invalid values. Expected only 'yes' or 'no'."
        )
        st.write("Unique values found:", df["y"].unique())
        st.stop()

    X = df.drop(columns=["y"])

    model_name = MODEL_OPTIONS[model_label]
    st.info(f"Using model: **{model_label}**")

    # Evaluate button
    if st.button("🚀 Evaluate Model"):

        with st.spinner("Running model evaluation..."):

            model, threshold = load_model_and_threshold(model_name)

            metrics, preds, probs = evaluate(
                model,
                threshold,
                X,
                y,
            )

        st.success("✅ Evaluation Complete")
        st.markdown("---")

        # Metrics
        st.subheader("📊 Evaluation Metrics")
        cols = st.columns(6)

        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k, f"{v:.4f}")

        st.markdown("---")

        # Confusion Matrix
        st.subheader("🧮 Confusion Matrix")

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y,
            preds,
            ax=ax,
            display_labels=["No Subscription", "Subscription"],
        )
        st.pyplot(fig)

        st.markdown("---")

        # ROC Curve
        st.subheader("📉 ROC Curve")

        try:
            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y, probs, ax=ax)
            st.pyplot(fig)
        except Exception:
            st.warning("ROC curve could not be generated.")

        st.markdown("---")

        # Classification Report
        st.subheader("📋 Classification Report")

        report_dict = classification_report(
            y,
            preds,
            output_dict=True,
            target_names=["No Subscription", "Subscription"],
        )

        report_df = pd.DataFrame(report_dict).transpose()

        if "accuracy" in report_df.index:
            report_df = report_df.drop(index="accuracy")

        report_df = report_df.round(3)
        st.dataframe(report_df)
        st.markdown("---")

        # Predictions Preview
        st.subheader("🔮 Predictions Preview")

        preview_df = X.copy()
        preview_df["Actual"] = y.values
        preview_df["Predicted"] = preds
        preview_df["Probability"] = probs

        st.dataframe(preview_df.head(20))

        # Download Predictions
        csv = preview_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Predictions",
            csv,
            "predictions.csv",
            "text/csv",
        )

# Footer
st.markdown("---")
st.caption("Developed for ML Assignment — Bank Marketing Classification")
