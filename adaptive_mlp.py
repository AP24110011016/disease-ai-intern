import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/diabetes.csv")

# Replace invalid zeros
cols_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in cols_with_zero:
    median = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median)

# -----------------------------
# Split into 10 Sequential Batches
# -----------------------------
batch_size = len(df) // 10

batches = []

for i in range(10):

    start = i * batch_size

    if i == 9:
        end = len(df)
    else:
        end = (i + 1) * batch_size

    batches.append(df.iloc[start:end])

# -----------------------------
# Adaptive Retraining
# -----------------------------
results = []

os.makedirs("results", exist_ok=True)

for i in range(1, len(batches)):

    print(f"\n========== Batch {i} ==========")

    train_df = pd.concat(batches[:i])

    test_df = batches[i]

    X_train = train_df.drop("Outcome", axis=1)
    y_train = train_df["Outcome"]

    X_test = test_df.drop("Outcome", axis=1)
    y_test = test_df["Outcome"]

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Adaptive MLP Model
    # -----------------------------
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(32, activation="relu"),
        Dropout(0.3),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        verbose=0
    )

    # -----------------------------
    # Prediction
    # -----------------------------
    probabilities = model.predict(X_test, verbose=0).flatten()

    threshold = 0.50

    predictions = (probabilities >= threshold).astype(int)
    
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, probabilities)

    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    results.append({
        "Batch": i,
        "Train Samples": len(train_df),
        "Test Samples": len(test_df),
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Threshold": threshold
    })

# -----------------------------
# Save Results
# -----------------------------
results_df = pd.DataFrame(results)

print("\n==============================")
print("Adaptive Training Results")
print("==============================")
print(results_df)

results_df.to_csv(
    "results/adaptive_results.csv",
    index=False
)

print("\nAdaptive training completed!")
print("Results saved to results/adaptive_results.csv")

import numpy as np

os.makedirs("results/roc_data", exist_ok=True)

np.savez(
    "results/roc_data/adaptive_mlp_roc.npz",
    fpr=fpr,
    tpr=tpr
)

print("Adaptive MLP ROC data saved successfully.")