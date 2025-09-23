# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score
)

# Import your helper function
from fraud_pipeline import predict_fraud

# ------------------------------
# Load pipeline bundle (models + encoders + threshold)
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
# Load Dataset (Default or User Upload)
# ------------------------------
st.sidebar.markdown("### 📂 Data Options")

uploaded_file = st.sidebar.file_uploader("Upload your CSV (optional)", type="csv")

if uploaded_file is not None:
    st.info("Using uploaded dataset")
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default demo dataset (~263 MB from Google Drive)")
    file_id = "1U90O4Mo4lCkVoY_0pYGwS_VbWFO59Po3"  # <-- replace with your file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    df = pd.read_csv(url)

# ------------------------------
# Show dataset info in sidebar
# ------------------------------
with st.sidebar.expander("📊 Dataset Info"):
    st.write("**Columns in dataset:**")
    st.write(list(df.columns))

    st.write("**First 5 rows:**")
    st.write(df.head())

# ------------------------------
# App Title
# ------------------------------
st.title("💳 Fraud Detection App")
st.write("Predict whether a transaction is FRAUD or LEGIT using XGBoost + LightGBM ensemble.")

# Sidebar: threshold adjustment
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, threshold, 0.01)

# ==============================
# Retrain Model Button (with progress + timer)
# ==============================
if st.sidebar.button("🔄 Retrain Models"):
    start_time = time.time()

    progress = st.progress(0)
    status = st.empty()

    # Step 1: Data preparation
    status.text("📂 Preparing data...")
    if uploaded_file is None:
        st.warning("⚡ Using a 50k-row sample of the demo dataset for retraining (to save time).")
        df_train = df.sample(n=50000, random_state=42)
    else:
        st.success("✅ Using full uploaded dataset for retraining.")
        df_train = df

    X = df_train.drop("is_fraud", axis=1)
    y = df_train["is_fraud"]

    progress.progress(25)

    # Step 2: Train XGBoost
    status.text("🌲 Training XGBoost model...")
    from xgboost import XGBClassifier
    new_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=5
    )
    new_xgb.fit(X, y)
    progress.progress(50)

    # Step 3: Train LightGBM
    status.text("⚡ Training LightGBM model...")
    from lightgbm import LGBMClassifier
    new_lgbm = LGBMClassifier(
        n_estimators=100,
        max_depth=-1
    )
    new_lgbm.fit(X, y)
    progress.progress(75)

    # Step 4: Save bundle
    status.text("💾 Saving updated pipeline...")
    new_bundle = {
        "xgb_model": new_xgb,
        "lgbm_model": new_lgbm,
        "encoders": encoders,
        "threshold": threshold
    }
    joblib.dump(new_bundle, "fraud_pipeline.pkl")
    progress.progress(100)

    elapsed = time.time() - start_time
    status.text("✅ Retraining complete!")
    st.success(f"Models retrained in {elapsed:.2f} seconds")
    st.balloons()
    
# ==============================
# Show Evaluation Metrics
# ==============================
if st.sidebar.button("📊 Show Model Evaluation"):
    st.subheader("Model Evaluation")

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Precompute ensemble
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    lgb_prob = lgbm_model.predict_proba(X_test)[:, 1]
    ensemble_prob = (xgb_prob + lgb_prob) / 2
    ensemble_pred = (ensemble_prob >= threshold).astype(int)

    # Model outputs
    model_results = {
        "XGBoost": (xgb_model.predict(X_test), xgb_prob),
        "LightGBM": (lgbm_model.predict(X_test), lgb_prob),
        "Ensemble": (ensemble_pred, ensemble_prob)
    }

    # --- Summary Table ---
    summary = []
    for model_name, (y_pred, y_prob) in model_results.items():
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = auc(fpr, tpr)

        summary.append({
            "Model": model_name,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
            "AUC": round(auc_val, 3)
        })

    summary_df = pd.DataFrame(summary).set_index("Model")

    st.markdown("### 📊 Metrics Summary")
    st.dataframe(summary_df)

    # --- Download button for summary ---
    csv_data = summary_df.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️ Download Metrics as CSV",
        data=csv_data,
        file_name="fraud_model_metrics.csv",
        mime="text/csv"
    )

    # --- Tabs for details ---
    tab_xgb, tab_lgbm, tab_ens = st.tabs(["XGBoost", "LightGBM", "Ensemble"])

    for tab, (model_name, (y_pred, y_prob)) in zip(
        [tab_xgb, tab_lgbm, tab_ens], model_results.items()
    ):
        with tab:
            st.markdown(f"### 🔹 {model_name}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"]
            )
            ax.set_title(f"{model_name} Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_title(f"{model_name} ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)

# ==============================
# Prediction UI
# ==============================
st.header("🔮 Predict a Single Transaction")

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
    result = predict_fraud(input_df, xgb_model, lgbm_model, encoders, threshold)
    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {result['prediction']}")
    st.write(f"**Fraud Probability:** {result['fraud_probability']:.4f}")
    st.write(f"**Threshold Used:** {result['threshold']:.2f}")

    if result['prediction'] == "FRAUD":
        st.error("🚨 Fraudulent transaction detected!")
        st.snow()
    else:
        st.success("✅ Transaction looks legit!")
        st.balloons()
