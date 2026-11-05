
import joblib
import pandas as pd
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load Day 1 model
day1_model = joblib.load(
    BASE_DIR / "models" / "churn_pipeline.pkl"
)

# Load Day 4 XGBoost model
day4_model = joblib.load(
    BASE_DIR / "models" / "churn_v4.joblib"
)


# Feature engineering for Day 4
def engineer_features(data):

    data = data.copy()

    data["tenure_group"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    ).astype("object")

    data["avg_monthly_spend"] = (
        data["TotalCharges"] / (data["tenure"] + 1)
    )

    data["is_month_to_month"] = (
        data["Contract"] == "Month-to-month"
    ).astype(int)

    data["has_fiber"] = (
        data["InternetService"] == "Fiber optic"
    ).astype(int)

    data["is_electronic_check"] = (
        data["PaymentMethod"] == "Electronic check"
    ).astype(int)

    data["has_online_security"] = (
        data["OnlineSecurity"] == "Yes"
    ).astype(int)

    return data


# Common prediction helper
def get_result(model, customer_data):

    df = pd.DataFrame([customer_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn": "Yes" if prediction == 1 else "No",
        "probability": float(probability)
    }


# Day 1 - Logistic Regression
def predict(customer_data):

    return get_result(day1_model, customer_data)


# Day 4 - XGBoost
def predict_xgb(customer_data):

    df = pd.DataFrame([customer_data])

    # Apply Day 4 feature engineering
    df = engineer_features(df)

    return get_result(day4_model, df)
