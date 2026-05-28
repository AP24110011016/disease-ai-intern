import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/students.csv")

# Features (inputs)
X = df[["Age", "StudyHours", "Marks"]]

# Target (output)
y = df["Pass"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print predictions
print("Predictions:")
print(predictions)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(accuracy)