import pandas as pd
import matplotlib.pyplot as plt
import os

# Create figures folder
os.makedirs("figures", exist_ok=True)

# Load results
df = pd.read_csv("results/ablation_results.csv")

# Create figure
plt.figure(figsize=(8,6))

x = range(len(df))
width = 0.25

plt.bar(
    [i - width for i in x],
    df["Precision"],
    width=width,
    label="Precision"
)

plt.bar(
    x,
    df["Recall"],
    width=width,
    label="Recall"
)

plt.bar(
    [i + width for i in x],
    df["F1"],
    width=width,
    label="F1 Score"
)

plt.xticks(
    x,
    df["Configuration"],
    rotation=10
)

plt.ylabel("Score")
plt.title("Ablation Study Comparison")

plt.ylim(0, 1)

plt.legend()

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "figures/Figure4_Ablation_Study.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to figures/Figure4_Ablation_Study.png")