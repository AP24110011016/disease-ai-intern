import pandas as pd

df = pd.read_csv("data/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 500]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

best_rf = grid.best_estimator_

y_pred = best_rf.predict(X_test)

y_prob = best_rf.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nRF Results")
print("Accuracy :", round(accuracy,4))
print("Precision:", round(precision,4))
print("Recall   :", round(recall,4))
print("F1 Score :", round(f1,4))
print("ROC AUC  :", round(roc_auc,4))

import matplotlib.pyplot as plt

importance = best_rf.feature_importances_

features = X.columns

plt.figure(figsize=(8,5))

plt.bar(features, importance)

plt.xticks(rotation=45)

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "results/rf_feature_importance.png"
)

plt.show()

new_row = pd.DataFrame([{
    "Model":"RF",
    "Accuracy":accuracy,
    "Precision":precision,
    "Recall":recall,
    "F1":f1,
    "ROC_AUC":roc_auc
}])

existing = pd.read_csv(
    "results/results_log.csv"
)

existing = pd.concat(
    [existing,new_row],
    ignore_index=True
)

existing.to_csv(
    "results/results_log.csv",
    index=False
)

print("RF results logged")

import numpy as np
import os

os.makedirs("results/roc_data", exist_ok=True)

np.savez(
    "results/roc_data/random_forest_roc.npz",
    fpr=fpr,
    tpr=tpr
)

print("Random Forest ROC data saved successfully.")