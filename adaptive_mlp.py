import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

    train_df = pd.concat(batches[:i])

    test_df = batches[i]

    X_train = train_df.drop("Outcome", axis=1)
    y_train = train_df["Outcome"]

    X_test = test_df.drop("Outcome", axis=1)
    y_test = test_df["Outcome"]

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
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

    probabilities = model.predict(X_test, verbose=0)

    threshold = 0.50

    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    results.append({
        "Batch": i,
        "Train Samples": len(train_df),
        "Test Samples": len(test_df),
        "Accuracy": round(accuracy, 4),
        "F1 Score": round(f1, 4),
        "Threshold": threshold
    })

# -----------------------------
# Save Results
# -----------------------------
results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv(
    "results/adaptive_results.csv",
    index=False
)

print("\nAdaptive training completed!")