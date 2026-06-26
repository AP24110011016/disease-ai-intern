import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/diabetes.csv")

# Replace invalid zeros with median
cols_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in cols_with_zero:
    median_value = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median_value)

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# Batch Size Experiments
# -----------------------------
batch_sizes = [32, 64, 128]

results = []

os.makedirs("results", exist_ok=True)

for batch_size in batch_sizes:

    print("\n" + "=" * 50)
    print(f"Training with Batch Size = {batch_size}")
    print("=" * 50)

    # Build Model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Learning Rate Scheduler
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=batch_size,
        callbacks=[reduce_lr],
        verbose=1
    )

    # Evaluate
    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"\nTest Accuracy: {accuracy:.4f}")

    results.append({
        "batch_size": batch_size,
        "accuracy": float(accuracy)
    })

# -----------------------------
# Results DataFrame
# -----------------------------
results_df = pd.DataFrame(results)

print("\nBatch Size Comparison")
print(results_df)

results_df.to_csv(
    "results/mlp_tuning_results.csv",
    index=False
)

# -----------------------------
# Find Best Configuration
# -----------------------------
best_config = max(
    results,
    key=lambda x: x["accuracy"]
)

print("\nBest Configuration")
print(best_config)

# -----------------------------
# Save Best Config
# -----------------------------
with open("best_config.json", "w") as f:
    json.dump(best_config, f, indent=4)

print("\nBest configuration saved to best_config.json")