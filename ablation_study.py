import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ---------------------------------
# Load Dataset
# ---------------------------------

df = pd.read_csv("data/diabetes.csv")

cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for c in cols:
    median = df[df[c] != 0][c].median()
    df[c] = df[c].replace(0, median)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# Same held-out test set for all models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

results = []

# ---------------------------------
# Function
# ---------------------------------

def build_model():

    model = Sequential([

        Dense(64,
              activation="relu",
              input_shape=(8,)),

        Dropout(0.3),

        Dense(32,
              activation="relu"),

        Dropout(0.3),

        Dense(1,
              activation="sigmoid")

    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ---------------------------------
# Configuration A
# Static MLP
# ---------------------------------

print("\nTraining Static MLP")

model = build_model()

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    verbose=0
)

pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int)

results.append({

    "Configuration":"Static MLP",

    "Precision":round(
        precision_score(y_test,pred),4),

    "Recall":round(
        recall_score(y_test,pred),4),

    "F1":round(
        f1_score(y_test,pred),4)

})

# ---------------------------------
# Configuration B
# Adaptive
# ---------------------------------

print("Training Adaptive MLP")

batch_size=len(X_train)//5

for i in range(5):

    end=(i+1)*batch_size

    model=build_model()

    model.fit(
        X_train[:end],
        y_train.iloc[:end],
        epochs=20,
        batch_size=32,
        verbose=0
    )

pred=(model.predict(X_test,verbose=0)>=0.5).astype(int)

results.append({

    "Configuration":"Adaptive",

    "Precision":round(
        precision_score(y_test,pred),4),

    "Recall":round(
        recall_score(y_test,pred),4),

    "F1":round(
        f1_score(y_test,pred),4)

})

# ---------------------------------
# Configuration C
# Full System
# ---------------------------------

print("Training Full System")

prob=model.predict(
    X_test,
    verbose=0
).flatten()

threshold=0.45

pred=(prob>=threshold).astype(int)

results.append({

    "Configuration":"Adaptive + Proactive",

    "Precision":round(
        precision_score(y_test,pred),4),

    "Recall":round(
        recall_score(y_test,pred),4),

    "F1":round(
        f1_score(y_test,pred),4)

})

# ---------------------------------
# Save Results
# ---------------------------------

results_df=pd.DataFrame(results)

print("\nAblation Study Results\n")

print(results_df)

os.makedirs("results",exist_ok=True)

results_df.to_csv(

    "results/ablation_results.csv",

    index=False

)

print("\nSaved to results/ablation_results.csv")