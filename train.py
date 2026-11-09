import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ==============================================================================
# 1. Dataset Simulation & Feature Engineering
# ==============================================================================
# Set deterministic seed for reproducibility
np.random.seed(42)
n_samples = 5000

# Generate synthetic applicant financials mirroring real-world credit distributions:
# - income: normal distribution around $65k
# - credit_score: normal distribution around 700 (FICO scale)
# - loan_amount: requested principal around $250k
# - years_employed: exponential distribution modeling job tenure
data = pd.DataFrame({
    'income': np.random.normal(65000, 20000, n_samples),
    'credit_score': np.random.normal(700, 50, n_samples),
    'loan_amount': np.random.normal(250000, 100000, n_samples),
    'years_employed': np.random.exponential(5, n_samples),
})

# Underwriting Policy Rule:
# Score components:
# - Credit score > 650 contributes 40%
# - Annual income > 20% of requested loan (debt-to-income benchmark) contributes 40%
# - Job tenure > 2 years contributes 20%
# Applications with an aggregate score > 0.60 are marked as Approved (1); else Denied (0).
data['approval_score'] = (
    (data['credit_score'] > 650).astype(int) * 0.4 +
    (data['income'] > data['loan_amount'] * 0.2).astype(int) * 0.4 +
    (data['years_employed'] > 2).astype(int) * 0.2
)
data['approved'] = (data['approval_score'] > 0.6).astype(int)
data = data.drop(columns=['approval_score'])

# Feature matrix (X) and target vector (y)
X = data[['income', 'credit_score', 'loan_amount', 'years_employed']]
y = data['approved']

# ==============================================================================
# 2. Data Partitioning: Train (80%), Validation (10%), Test (10%)
# ==============================================================================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set:   {X_train.shape[0]} rows")
print(f"Validation set: {X_val.shape[0]} rows")
print(f"Test set:       {X_test.shape[0]} rows")

# ==============================================================================
# 3. Model Training: Gradient Boosting Classifier
# ==============================================================================
# Tuned hyperparameters for conservative, calibrated probability estimates
gb_model = GradientBoostingClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=4, 
    random_state=42
)
gb_model.fit(X_train, y_train)

# ==============================================================================
# 4. Validation Set Performance Evaluation
# ==============================================================================
val_preds = gb_model.predict(X_val)
print("\n--- Validation Set Performance ---")
print(classification_report(y_val, val_preds))

# ==============================================================================
# 5. Final Holdout Test Set Evaluation & ROC-AUC Metric
# ==============================================================================
test_preds = gb_model.predict(X_test)
test_probs = gb_model.predict_proba(X_test)[:, 1]
print("\n--- Test Set Performance ---")
print(classification_report(y_test, test_preds))
print(f"Test ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")

# ==============================================================================
# 6. Artifact Serialization for Production Inference
# ==============================================================================
joblib.dump(gb_model, 'loan_gradient_boosting_model.pkl')
print("\nModel saved successfully as 'loan_gradient_boosting_model.pkl'!")
