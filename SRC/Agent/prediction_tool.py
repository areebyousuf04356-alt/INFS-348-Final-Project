
import joblib
import pandas as pd
from pathlib import Path


# Locate the project directory and saved model
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "Model" / "fraud_xgboost_model.joblib"

# Load the trained model bundle
model_bundle = joblib.load(MODEL_PATH)

fraud_model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]
threshold = model_bundle["threshold"]
feature_columns = model_bundle["feature_columns"]


def predict_transaction(transaction):
    """
    Predict whether a credit card transaction is fraudulent.

    Args:
        transaction (dict): Transaction data containing
                            the 24 expected feature values.

    Returns:
        dict: Fraud probability, classification, and threshold.
    """

    # Check for missing features
    missing_features = [
        feature for feature in feature_columns
        if feature not in transaction
    ]

    if missing_features:
        raise ValueError(
            f"Missing transaction features: {missing_features}"
        )

    # Arrange transaction features in the expected order
    transaction_df = pd.DataFrame(
        [[transaction[column] for column in feature_columns]],
        columns=feature_columns
    )

    # Apply the saved preprocessing
    processed_transaction = preprocessor.transform(transaction_df)

    # Predict fraud probability
    fraud_probability = float(
        fraud_model.predict_proba(processed_transaction)[0, 1]
    )

    # Apply the selected classification threshold
    prediction = (
        "Fraudulent"
        if fraud_probability >= threshold
        else "Legitimate"
    )

    return {
        "prediction": prediction,
        "fraud_probability": round(fraud_probability, 4),
        "threshold": float(threshold)
    }
