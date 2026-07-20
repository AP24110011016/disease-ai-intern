import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import os

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Replace invalid zeros with column median
cols_with_zero = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

for col in cols_with_zero:
    median_value = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median_value)

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)
# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_prob)

# Results dataframe
results = pd.DataFrame({
    "model": ["Logistic Regression"],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1],
    "auc_roc": [auc_roc]
})

# Create results folder if needed
os.makedirs("results", exist_ok=True)

# Save results
results.to_csv("results/results_log.csv", index=False)

print(results)
print("\nResults saved to results/results_log.csv")

import numpy as np
import os

os.makedirs("results/roc_data", exist_ok=True)

np.savez(
    "results/roc_data/logistic_regression_roc.npz",
    fpr=fpr,
    tpr=tpr
)

print("ROC data saved successfully.")