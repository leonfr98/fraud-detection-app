#app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score
)

# Import pipeline function
from fraud_pipeline import predict_fraud

# ------------------------------
# Load trained pipeline
# ------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("fraud_pipeline.pkl")

bundle = load_pipeline()
xgb_model = bundle["xgb_model"]
lgbm_model = bundle["lgbm_model"]
encoders  = bundle["encoders"]
threshold = bundle["threshold"]

# ------------------------------
# Tabs: About | Prediction | Metrics
# ------------------------------
tab_about, tab_predict, tab_metrics = st.tabs(["ℹ️ About", "🔮 Prediction", "📊 Metrics"])

# ------------------------------
# About Tab
# ------------------------------
with tab_about:
    st.title("💳 Fraud Detection App")
    st.write("XGBoost + LightGBM Ensemble Model")

    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        st.markdown(readme_content, unsafe_allow_html=True)
    else:
        st.info("README.md not found. Please check the repository.")

# ------------------------------
# Prediction Tab
# ------------------------------
with tab_predict:
    st.header("🔮 Single Transaction Prediction")

    # Sidebar: threshold adjustment
    threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, threshold, 0.01)

    amt = st.number_input("Transaction Amount", min_value=0.0, value=120.0)
    category = st.selectbox("Category", [
        "entertainment", "grocery_net", "grocery_pos",
        "gas_transport", "shopping_net", "shopping_pos", "travel"
    ])
    isMan = st.selectbox("Is Customer Male?", [0, 1])
    isLateEvening = st.selectbox("Is Late Evening?", [0, 1])
    city_pop = st.number_input("City Population", min_value=0, value=50000)
    job = st.selectbox("Job", ["Administrator", "Barrister", "Electrical engineer"])
    lat = st.number_input("Latitude", value=37.5)
    long = st.number_input("Longitude", value=-122.0)
    secs_since_prev = st.number_input("Seconds Since Previous Transaction", min_value=0, value=45)
    tx_count_1h = st.number_input("Transactions in Last 1h", min_value=0, value=2)
    tx_count_24h = st.number_input("Transactions in Last 24h", min_value=0, value=10)

    input_df = pd.DataFrame([{
        "amt": amt,
        "category": category,
        "isMan": isMan,
        "isLateEvening": isLateEvening,
        "city_pop": city_pop,
        "job": job,
        "lat": lat,
        "long": long,
        "secs_since_prev": secs_since_prev,
        "tx_count_1h": tx_count_1h,
        "tx_count_24h": tx_count_24h
    }])

    if st.button("🚀 Run Prediction"):
        result = predict_fraud(input_df, xgb_model, lgbm_model, encoders, threshold).iloc[0]

        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {result['prediction']}")
        st.write(f"**Fraud Probability:** {result['fraud_probability']:.4f}")
        st.write(f"**Threshold Used:** {result['threshold']:.2f}")

        # Animations
        if result['prediction'] == "FRAUD":
            st.error("🚨 Fraudulent transaction detected!")
            st.snow()
        else:
            st.success("✅ Transaction looks legit!")
            st.balloons()

# ------------------------------
# Metrics Tab
# ------------------------------
with tab_metrics:
    st.header("📊 Model Performance Metrics")

    metrics_file = st.file_uploader("Upload a labeled CSV for evaluation (must include target column)", type="csv")

    if metrics_file is not None:
        df_eval = pd.read_csv(metrics_file)

        # Auto-detect target column
        possible_targets = ["label", "fraud", "is_fraud", "target"]
        target_col = None
        for col in possible_targets:
            if col in df_eval.columns:
                target_col = col
                break

        if target_col is None:
            st.error("❌ Could not find target column. Please include one of: " + ", ".join(possible_targets))
        else:
            X_eval = df_eval.drop(target_col, axis=1)
            y_true = df_eval[target_col]

            # Vectorized predictions
            results = predict_fraud(X_eval, xgb_model, lgbm_model, encoders, threshold)
            y_pred = (results["prediction"] == "FRAUD").astype(int)
            y_prob = results["fraud_probability"]

            # Metrics
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = auc(fpr, tpr)

            st.write("**Precision:**", round(precision, 3))
            st.write("**Recall:**", round(recall, 3))
            st.write("**F1 Score:**", round(f1, 3))
            st.write("**AUC:**", round(auc_val, 3))

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Legit", "Fraud"],
                        yticklabels=["Legit", "Fraud"])
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Classification Report
            st.text("Classification Report:")
            st.text(classification_report(y_true, y_pred))

            # Download results
            out_df = X_eval.copy()
            out_df["y_true"] = y_true
            out_df["fraud_probability"] = y_prob
            out_df["prediction"] = results["prediction"]

            csv_out = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )
