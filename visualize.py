import pandas as pd

# Load dataset
df = pd.read_csv("data/students.csv")

# First 5 rows
print("First 5 Rows:")
print(df.head())

# Last 5 rows
print("\nLast 5 Rows:")
print(df.tail())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Select one column
print("\nMarks Column:")
print(df["Marks"])

# Select multiple columns
print("\nName and Marks:")
print(df[["Name", "Marks"]])

# Students with marks above 88
print("\nStudents with Marks > 88:")
print(df[df["Marks"] > 88])

import matplotlib.pyplot as plt

# Histogram
plt.hist(df["Marks"])

# Labels
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.title("Marks Distribution")

# Show graph
plt.show()

# Scatter plot
plt.scatter(df["Age"], df["Marks"])

plt.xlabel("Age")
plt.ylabel("Marks")
plt.title("Age vs Marks")

plt.show()