#app.py
import streamlit as st
import pandas as pd
import joblib

# Import your functions
from fraud_pipeline import predict_fraud

# Load pipeline bundle (models + encoders + threshold)
@st.cache_resource
def load_pipeline():
    return joblib.load("fraud_pipeline.pkl")

bundle = load_pipeline()
xgb_model = bundle["xgb_model"]
lgbm_model = bundle["lgbm_model"]
encoders  = bundle["encoders"]
threshold = bundle["threshold"]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("💳 Fraud Detection App")
st.write("Predict whether a transaction is FRAUD or LEGIT using XGBoost + LightGBM ensemble.")

# Sidebar: threshold adjustment
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, threshold, 0.01)

st.header("Enter Transaction Data")

# Input fields (you can add more based on your features)
amt = st.number_input("Transaction Amount", min_value=0.0, value=120.0)
category = st.selectbox("Category", ["entertainment", "grocery_net", "grocery_pos","gas_transport", "shopping_net","shopping_pos", "travel"])
isMan = st.selectbox("Is Customer Male?", [0, 1])
isLateEvening = st.selectbox("Is Late Evening?", [0, 1])
city_pop = st.number_input("City Population", min_value=0, value=50000)
job = st.selectbox("Job", ["Administrator","Barrister","Electrical engineer"])
lat = st.number_input("Latitude", value=37.5)
long = st.number_input("Longitude", value=-122.0)
secs_since_prev = st.number_input("Seconds Since Previous Transaction", min_value=0, value=45)
tx_count_1h = st.number_input("Transactions in Last 1h", min_value=0, value=2)
tx_count_24h = st.number_input("Transactions in Last 24h", min_value=0, value=10)

# Prepare input dataframe
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

# Prediction button
if st.button("🔮 Predict"):
    result = predict_fraud(input_df, xgb_model, lgbm_model, encoders, threshold)
    st.subheader("Result")
    st.write(f"**Prediction:** {result['prediction']}")
    st.write(f"**Fraud Probability:** {result['fraud_probability']:.4f}")
    st.write(f"**Threshold Used:** {result['threshold']:.2f}")
    
    if result['prediction'] == "FRAUD":
        st.error("🚨 Fraudulent transaction detected!")
        st.snow()
    else:
        st.success("✅ Transaction looks legit!")
        st.balloons()
