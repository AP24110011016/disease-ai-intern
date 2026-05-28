import pandas as pd

# Load dataset
df = pd.read_csv("data/students.csv")

# Print dataset
print("Original Dataset:")
print(df)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

# Print cleaned dataset
print("\nCleaned Dataset:")
print(df)

# Sort by Marks
sorted_df = df.sort_values(by="Marks", ascending=False)

print("\nSorted Dataset:")
print(sorted_df)

# Count city frequency
print("\nCity Counts:")
print(df["City"].value_counts())

# Correlation
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)

plt.title("Correlation Heatmap")
plt.show()

# Boxplot
sns.boxplot(y=df["Marks"])

plt.title("Marks Boxplot")
plt.show()