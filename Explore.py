import pandas as pd

# Load CSV file
df = pd.read_csv("data/students.csv")

# Print complete dataset
print("Dataset:")
print(df)

# Print shape
print("\nShape:")
print(df.shape)

# Print data types
print("\nData Types:")
print(df.dtypes)

# Print statistical summary
print("\nStatistics:")
print(df.describe())