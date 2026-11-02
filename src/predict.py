
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("models/churn_pipeline.pkl")


def predict(customer_data):

    df = pd.DataFrame([customer_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn": "Yes" if prediction == 1 else "No",
        "probability": float(probability)
    }
