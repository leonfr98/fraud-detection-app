# Fraud Detection App

- This repository demonstrates fraud detection model in credit card transaction.
- For a model we are using first 1.048M transactions of this dataset from kaggle: https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset 
      (out of 1.85M in total)
- The model is an ensemble of XGBoost and Light GBM models

---

## 🚀 Live Demo
👉 [Streamlit Cloud App](https://fraud-detection-app-bxejrzq5zjleniz8ktbnfa.streamlit.app/)  

---

## 🖥 Running Locally
Clone the repo and install requirements:



## 📂 Test Data File Structure:

Column Descriptions

amt: Transaction amount

category: Transaction type (Valid values to use: "entertainment", "grocery_net", "grocery_pos","gas_transport", "shopping_net","shopping_pos", "travel")

isMan: 1 if customer is male, 0 if customer is a female

isLateEvening: 1 if transaction is late evening (between 10 p. m. and 11:59 p. m.), 0 otherwise

city_pop: City population of transaction location

job: Customer’s job (Valid values to use : "Administrator","Barrister","Electrical engineer")

lat, long: Transaction location coordinates

secs_since_prev: Seconds since customer’s last transaction

tx_count_1h: Number of transactions in the last 1 hour

tx_count_24h: Number of transactions in the last 24 hours

is_fraud: Target (0 = Legit, 1 = Fraud) → only required for evaluation, not for single predictions
