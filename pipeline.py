import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Columns where 0 indicates missing values
cols_with_zero = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

# Replace zeros with median
for col in cols_with_zero:
    median_value = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median_value)

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Min-Max Normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 80/20 Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Pipeline completed successfully!")