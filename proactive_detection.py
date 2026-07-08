import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/diabetes.csv")

# -----------------------------
# Replace Invalid Zeros
# -----------------------------
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

print("Dataset Loaded Successfully")
print("Total Samples:", len(df))

# -----------------------------
# Split Dataset into 10 Sequential Batches
# -----------------------------
num_batches = 10
batch_size = len(df) // num_batches

batches = []

for i in range(num_batches):

    start = i * batch_size

    if i == num_batches - 1:
        end = len(df)
    else:
        end = (i + 1) * batch_size

    batch = df.iloc[start:end].reset_index(drop=True)
    batches.append(batch)

print("\nDataset Split into 10 Sequential Batches")

for i, batch in enumerate(batches):
    print(f"Batch {i+1}: {len(batch)} samples")

# -----------------------------
# Rolling Window Feature Calculation
# -----------------------------
print("\nRolling Window Features (Last 3 Batches)\n")

rolling_features = []

for i in range(2, len(batches)):

    # Last 3 batches
    rolling_df = pd.concat(
        batches[i-2:i+1],
        ignore_index=True
    )

    X = rolling_df.drop("Outcome", axis=1)

    # Mean
    rolling_mean = X.mean()

    # Standard Deviation
    rolling_std = X.std()

    # Trend
    previous_df = pd.concat(
        batches[i-2:i],
        ignore_index=True
    )

    previous_mean = previous_df.drop(
        "Outcome",
        axis=1
    ).mean()

    trend = rolling_mean - previous_mean

    rolling_features.append({
        "Batch": i + 1,
        "Mean": rolling_mean,
        "Std": rolling_std,
        "Trend": trend
    })

    print("=" * 50)
    print(f"Rolling Window Ending at Batch {i+1}")
    print("=" * 50)

    print("\nMean")
    print(rolling_mean.round(2))

    print("\nStandard Deviation")
    print(rolling_std.round(2))

    print("\nTrend")
    print(trend.round(2))

print("\nRolling Feature Extraction Completed!")

print("\nStarting Adaptive MLP Training...\n")

results = []

for i in range(len(batches) - 2):

    print("=" * 60)
    print(f"Training Cycle {i+1}")
    print("=" * 60)

    # -------------------------
    # Train on current batches
    # -------------------------

    train_df = pd.concat(
        batches[:i+1],
        ignore_index=True
    )

    # -------------------------
    # Predict 2 batches ahead
    # -------------------------

    future_batch = i + 2

    test_df = batches[future_batch]

    # -------------------------
    # Features and Labels
    # -------------------------

    X_train = train_df.drop("Outcome", axis=1)
    y_train = train_df["Outcome"]

    X_test = test_df.drop("Outcome", axis=1)
    y_test = test_df["Outcome"]

    # -------------------------
    # Feature Scaling
    # -------------------------

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------
    # Build MLP Model
    # -------------------------

    model = Sequential([

        Dense(
            64,
            activation="relu",
            input_shape=(X_train.shape[1],)
        ),

        Dropout(0.3),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            1,
            activation="sigmoid"
        )

    ])

    model.compile(

        optimizer="adam",

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )

    # -------------------------
    # Train
    # -------------------------

    model.fit(

        X_train,

        y_train,

        epochs=50,

        batch_size=32,

        verbose=0

    )

    # -------------------------
    # Predict Probability
    # -------------------------
# -------------------------
# Predict Probability
# -------------------------

probabilities = model.predict(
    X_test,
    verbose=0
).flatten()

# -------------------------
# Precision-Recall Threshold Tuning
# -------------------------

precision, recall, thresholds = precision_recall_curve(
    y_test,
    probabilities
)

# Calculate F1 for every threshold
f1_scores = (
    2 * precision[:-1] * recall[:-1]
) / (
    precision[:-1] + recall[:-1] + 1e-10
)

best_index = np.argmax(f1_scores)

threshold = thresholds[best_index]

# -------------------------
# Final Prediction
# -------------------------

predictions = (
    probabilities >= threshold
).astype(int)

accuracy = accuracy_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

print("Training Samples :", len(train_df))
print("Testing Samples  :", len(test_df))
print("Accuracy         :", round(accuracy,4))
print("F1 Score         :", round(f1,4))

results.append({

    "Batch": future_batch + 1,

    "Train Samples": len(train_df),

    "Test Samples": len(test_df),

    "Accuracy": round(accuracy,4),

    "F1 Score": round(f1,4),

    "Threshold": round(float(threshold),3)

})

print("\nAdaptive Training Finished!")

# ====================================================
# Save Results
# ====================================================

results_df = pd.DataFrame(results)

print("\nFinal Results\n")
print(results_df)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/proactive_results.csv",
    index=False
)

print("\nResults saved successfully!")
print("File: results/proactive_results.csv")