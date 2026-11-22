# INFS-348-Final-Project
# Credit Card Fraud Detection & Explainability Agent

## 1. Project Overview

This project develops an AI-powered credit card fraud detection and explainability agent. It combines machine learning, explainable AI, and a large language model to analyze credit card transactions, estimate fraud probability, and communicate the factors influencing each prediction.

The system uses:
- **XGBoost** to classify transactions as fraudulent or legitimate.
- **SHAP** to identify the features contributing to each prediction.
- **OpenAI API** to generate clear, natural-language explanations.
- **Python** to coordinate the prediction, explanation, and language model components.

## 2. Project Objectives

- Develop a machine learning model to detect potentially fraudulent transactions.
- Evaluate the model using appropriate classification metrics.
- Explain individual predictions using SHAP.
- Build an AI agent that coordinates the prediction and explanation tools.
- Communicate model results in a clear and understandable format.

## 3. Dataset

The project uses `credit_card_fraud_2026.csv`, containing:

- 20,000 transactions.
- 26 columns, including 24 transaction features, a transaction identifier, and the target variable.
- 339 fraudulent transactions.
- 19,661 legitimate transactions.

The target variable is `is_fraud`. The transaction identifier is excluded from model training.

## 4. System Architecture

The agent follows a sequential workflow:

1. **Transaction Input:** Receives a transaction containing the 24 expected features.
2. **Prediction Tool:** Loads the trained XGBoost model and preprocessing pipeline to calculate fraud probability and classification.
3. **Explanation Tool:** Uses SHAP to identify the most influential features and their contributions.
4. **AI Agent:** Combines the prediction and SHAP results and sends them to an OpenAI language model.
5. **Natural-Language Output:** Returns an understandable assessment of the prediction and its contributing factors.

The trained model, prediction tool, explanation tool, and language model remain separate components.

## 5. Repository Structure

```text
credit-card-fraud-agent/
│
├── data/
│       └── credit_card_fraud_2026.csv
│
├── Model/
│   └── fraud_xgboost_model.joblib
│
├── SRC/
│   └── Agent/
│       ├── prediction_tool.py
│       ├── explanation_tool.py
│       └── agent.py
│
├── .gitignore
├── Requirements.txt
├── README.md
│
├── Data_Processing Notebook
├── Model_Training Notebook
└── Fraud_Agents Notebook
```

## 6. Installation and Setup

### Requirements

The project uses Python and the dependencies listed in `Requirements.txt`.

To install the required packages:

```bash
pip install -r Requirements.txt
```

### Google Colab Setup

1. Download or clone the repository and place the project folder in Google Drive.
2. Open `Fraud_Agents.ipynb` using Google Colab.
3. Mount Google Drive.
4. Set `PROJECT_ROOT` to the location of the project folder.
5. Run the dependency installation cell to install the pinned packages.

### OpenAI API Configuration

The AI agent requires an OpenAI API key to generate natural-language explanations.

For Google Colab:

1. Open the notebook's Secrets panel.
2. Add a secret named `OPENAI_API_KEY`.
3. Enter your API key and enable notebook access.
4. Load the secret into the environment:

```python
import os
from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

The agent accesses the key through an environment variable. **Never place API keys or passwords directly in source code or commit them to GitHub.**

## 7. Running the Demo

The primary demonstration is provided in `Fraud_Agents.ipynb`.

Run the notebook cells in order to:

1. Install dependencies and configure the environment.
2. Load the project files and trained model.
3. Import the AI agent.
4. Load the transaction dataset.
5. Test fraudulent and legitimate transactions.
6. Run randomized tests containing both transaction classes.
7. Review fraud probabilities, classifications, SHAP contributions, and AI-generated explanations.

The agent can also be imported into another Python script:

```python
from agent import analyze_transaction

result = analyze_transaction(transaction_data)

print(result)
```

Here, `transaction_data` must be a dictionary containing all 24 expected transaction features.

## 8. Model Evaluation

The XGBoost model was evaluated on a held-out test set using a classification threshold selected from the validation set.

| Metric | Test Result |
|---|---:|
| Precision | 0.2464 |
| Recall | 0.5000 |
| F1-score | 0.3301 |
| PR-AUC | 0.2714 |
| ROC-AUC | 0.9389 |

The classification threshold is approximately **0.6937**.

The model detected 17 of the 34 fraudulent transactions in the test set. It also produced 52 false positives, demonstrating that further refinement is needed.

## 9. Limitations and Responsible AI

- The dataset is highly imbalanced, with fraudulent transactions representing approximately 1.7% of the data.
- The model can produce false positives and false negatives.
- SHAP explains model contributions but does not establish the actual cause of a transaction.
- The language model communicates model outputs; it does not independently verify whether a transaction is fraudulent.
- Fraud predictions are estimates and should not be treated as definitive proof.
- Human review should remain part of consequential fraud investigation decisions.

## 10. Reproducibility and Security

- Dependency versions are pinned in `Requirements.txt`.
- The trained model and preprocessing pipeline are saved in the `Model` directory.
- The agent's components are organized in the `SRC/Agent` directory.
- API credentials are supplied through environment variables or Colab Secrets.
- API keys and passwords must never be committed to the repository.



