import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Create figures folder
os.makedirs("figures", exist_ok=True)

# Count classes
counts = df["Outcome"].value_counts().sort_index()

labels = ["Non-Diabetic", "Diabetic"]

plt.figure(figsize=(6,5))

bars = plt.bar(labels, counts)

plt.title("Class Distribution of Diabetes Dataset")
plt.ylabel("Number of Patients")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height()+5,
        str(int(bar.get_height())),
        ha="center",
        fontsize=12
    )

plt.tight_layout()

plt.savefig(
    "figures/dataset_distribution.png",
    dpi=300
)

plt.show()

print("Dataset distribution figure saved.")