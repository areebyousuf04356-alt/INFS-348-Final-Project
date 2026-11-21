
import os
from openai import OpenAI

from prediction_tool import predict_transaction
from explanation_tool import explain_transaction


# Initialize OpenAI client using an environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL_NAME = "gpt-6-luna"


def analyze_transaction(transaction):
    """
    Analyze a credit card transaction using XGBoost,
    SHAP, and an OpenAI language model.

    Returns:
        dict: Structured prediction, SHAP explanation,
              and natural-language assessment.
    """

    # Step 1: Predict fraud probability and classification
    prediction_result = predict_transaction(transaction)

    # Step 2: Generate SHAP explanation
    explanation_result = explain_transaction(transaction)

    # Step 3: Combine the analytical results
    analysis = {
        "prediction": prediction_result["prediction"],
        "fraud_probability": prediction_result["fraud_probability"],
        "classification_threshold": prediction_result["threshold"],
        "top_contributing_features": explanation_result["top_features"]
    }

    # Step 4: Prepare a concise prompt for the LLM
    instructions = """
    You are a credit card fraud analysis assistant.

    Explain the model's prediction using the supplied probability,
    classification threshold, and SHAP feature contributions.

    Requirements:
    - Respect the XGBoost model's classification.
    - Explain which features increase or decrease fraud likelihood.
    - Preserve the original meaning of feature names.
    - Do not invent transaction details or causes.
    - Explain that predictions are estimates, not proof of fraud.
    - Use clear, professional language.
    - Keep the response between 3 and 5 sentences.
    """

    # Step 5: Generate the natural-language assessment
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=instructions,
        input=str(analysis),
        max_output_tokens=200
    )

    # Step 6: Add the LLM assessment to the results
    analysis["explanation"] = response.output_text

    return analysis

