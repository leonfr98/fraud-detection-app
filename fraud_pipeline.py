#fraud_pipeline.py
import pandas as pd
import numpy as np

def predict_fraud(new_data: pd.DataFrame, xgb_model, lgbm_model, encoders, threshold=0.75):
    """
    Predict fraud probability and class using XGBoost + LightGBM ensemble.
    Works on single row or multiple rows (vectorized).
    """

    # ✅ 1. Standardize column names
    rename_map = {"t_count_1h": "tx_count_1h", "t_count_24h": "tx_count_24h"}
    new_data = new_data.rename(columns=rename_map).copy()

    # ✅ 2. Enforce expected feature order
    feature_order = [
        'amt', 'category', 'isMan', 'isLateEvening',
        'city_pop', 'job', 'lat', 'long',
        'secs_since_prev', 'tx_count_1h', 'tx_count_24h'
    ]
    for col in feature_order:
        if col not in new_data.columns:
            new_data[col] = 0

    # ✅ 3. Encode categoricals safely (vectorized)
    for col, le in encoders.items():
        if col in new_data.columns:
            values = new_data[col].astype(str)
            known_classes = set(le.classes_)
            safe_vals = values.where(values.isin(known_classes), 
                                     "Other" if "Other" in known_classes else values)
            new_data[col] = le.transform(safe_vals)

    # ✅ 4. Align with model's feature order
    xgb_feature_order = xgb_model.get_booster().feature_names
    new_data = new_data[xgb_feature_order]

    # ✅ 5. Model predictions
    xgb_prob = xgb_model.predict_proba(new_data)[:, 1]
    lgb_prob = lgbm_model.predict_proba(new_data)[:, 1]
    ensemble_prob = (xgb_prob + lgb_prob) / 2
    predictions = (ensemble_prob >= threshold).astype(int)

    # ✅ Return DataFrame with results (vectorized)
    return pd.DataFrame({
        "fraud_probability": ensemble_prob,
        "prediction": np.where(predictions == 1, "FRAUD", "LEGIT"),
        "threshold": threshold
    }, index=new_data.index)
