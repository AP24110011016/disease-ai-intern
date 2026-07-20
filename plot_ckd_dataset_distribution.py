import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Load CKD Dataset
# ----------------------------

file_path = "data/chronic_kidney_disease_full.arff"

with open(file_path, "r") as f:
    lines = f.readlines()

columns = [
    "age","bp","sg","al","su","rbc","pc","pcc","ba",
    "bgr","bu","sc","sod","pot","hemo","pcv",
    "wc","rc","htn","dm","cad","appet","pe","ane","class"
]

data = []

start = False

for line in lines:

    line = line.strip()

    if not start:
        if line.lower() == "@data":
            start = True
        continue

    if line == "" or line.startswith("%"):
        continue

    if line.endswith(","):
        line = line[:-1]

    values = [x.strip() for x in line.split(",")]

    if len(values) != 25:
        continue

    data.append(values)

df = pd.DataFrame(data, columns=columns)

# ----------------------------
# Missing Values
# ----------------------------

df.replace("?", np.nan, inplace=True)
df.replace("\t?", np.nan, inplace=True)
df.replace(" ?", np.nan, inplace=True)

for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

df.replace("nan", np.nan, inplace=True)

numeric_cols = [
    "age","bp","sg","al","su",
    "bgr","bu","sc","sod","pot",
    "hemo","pcv","wc","rc"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

num_cols = df.select_dtypes(include=["float64","int64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

# ----------------------------
# Encode Target
# ----------------------------

encoder = LabelEncoder()

df["class"] = encoder.fit_transform(df["class"])

# ----------------------------
# Plot
# ----------------------------

counts = df["class"].value_counts().sort_index()

labels = ["Class 0", "Class 1"]

os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(6,5))

bars = plt.bar(labels, counts)

plt.title("Class Distribution of CKD Dataset")
plt.ylabel("Number of Patients")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height()+3,
        str(int(bar.get_height())),
        ha="center",
        fontsize=12
    )

plt.tight_layout()

plt.savefig(
    "figures/ckd_dataset_distribution.png",
    dpi=300
)

plt.show()

print("CKD dataset distribution figure saved.")