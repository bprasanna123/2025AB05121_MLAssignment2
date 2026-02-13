import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Bank Customer Churn Prediction")
st.caption("Machine Learning Assignment – Streamlit Deployment")

st.info(
    "Upload a dataset with the SAME structure as the training dataset "
    "(Bank Customer Churn dataset). This app demonstrates ML deployment."
)


# ============================================
# LOAD MODELS & SCALER
# ============================================
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl"),
    }

    scaler = joblib.load("scaler.pkl")
    return models, scaler


models, scaler = load_models()


# ============================================
# FILE UPLOAD
# ============================================
st.header("📂 Upload CSV Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file (test dataset recommended)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ============================================
    # MODEL SELECTION
    # ============================================
    st.subheader("Model Selection")

    model_name = st.selectbox(
        "Choose Classification Model",
        list(models.keys())
    )

    model = models[model_name]

    # ============================================
    # PREPROCESSING (CRUCIAL FIX)
    # ============================================

    # Drop non-informative columns (same as training)
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Separate target if present
    if "Exited" in df.columns:
        y_true = df["Exited"]
        X = df.drop("Exited", axis=1)
    else:
        y_true = None
        X = df

    # One-hot encoding (same method as training)
    X = pd.get_dummies(X, drop_first=True)

    # Match training feature columns exactly
    training_features = scaler.feature_names_in_
    X = X.reindex(columns=training_features, fill_value=0)

    # Feature scaling
    X_scaled = scaler.transform(X)

    # ============================================
    # PREDICTIONS
    # ============================================
    predictions = model.predict(X_scaled)

    st.subheader("Prediction Output")
    st.write(predictions)

    # ============================================
    # METRICS (IF TARGET AVAILABLE)
    # ============================================
    if y_true is not None:

        st.subheader("Model Evaluation Metrics")

        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        mcc = matthews_corrcoef(y_true, predictions)

        if hasattr(model, "predict_proba"):
            auc = roc_auc_score(
                y_true,
                model.predict_proba(X_scaled)[:, 1]
            )
        else:
            auc = "N/A"

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")

        col1.metric("F1 Score", f"{f1:.4f}")
        col2.metric("MCC", f"{mcc:.4f}")
        col3.metric("AUC", auc)

        # ============================================
        # CONFUSION MATRIX
        # ============================================
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, predictions)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ============================================
        # CLASSIFICATION REPORT
        # ============================================
        st.subheader("Classification Report")
        report_df = pd.DataFrame(
            classification_report(y_true, predictions, output_dict=True)
            ).transpose()

        st.dataframe(report_df)


else:
    st.warning("Please upload a CSV dataset to proceed.")
