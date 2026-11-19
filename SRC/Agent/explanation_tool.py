
import shap
import numpy as np

from prediction_tool import fraud_model, preprocessor, feature_columns


# Create the SHAP explainer using the trained XGBoost model
explainer = shap.TreeExplainer(fraud_model)


def explain_transaction(transaction, top_n=5):
    """
    Explain the model's fraud prediction using SHAP.

    Args:
        transaction (dict): Transaction data containing the
                            24 expected feature values.
        top_n (int): Number of most influential features to return.

    Returns:
        dict: The most influential features and their SHAP values.
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
    import pandas as pd

    transaction_df = pd.DataFrame(
        [[transaction[column] for column in feature_columns]],
        columns=feature_columns
    )

    # Apply the saved preprocessing
    processed_transaction = preprocessor.transform(transaction_df)

    # Calculate SHAP values
    shap_values = explainer.shap_values(processed_transaction)

    # Handle the SHAP output as a numeric array
    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 1]
    else:
        shap_values = shap_values.reshape(-1)

    # Retrieve the names of the processed features
    feature_names = preprocessor.get_feature_names_out()

    # Rank features by absolute SHAP value
    contributions = sorted(
        zip(feature_names, shap_values),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    # Return the top contributing features
    explanation = []

    for feature, value in contributions[:top_n]:
        explanation.append({
            "feature": feature,
            "shap_value": round(float(value), 4),
            "impact": (
                "Increases fraud likelihood"
                if value > 0
                else "Decreases fraud likelihood"
            )
        })

    return {
        "top_features": explanation
    }
