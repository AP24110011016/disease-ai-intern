import pandas as pd
import matplotlib.pyplot as plt
import os

# Create figures folder
os.makedirs("figures", exist_ok=True)

# Load adaptive training results
df = pd.read_csv("results/adaptive_results.csv")

# Create figure
plt.figure(figsize=(8,6))

plt.plot(
    df["Train Samples"],
    df["Accuracy"],
    marker="o",
    linewidth=2,
    markersize=7
)

# Labels
plt.title("Adaptive Model Accuracy During Incremental Training", fontsize=14)
plt.xlabel("Training Samples", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)

plt.xticks(df["Train Samples"])

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "figures/Figure3_Adaptation_Accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to figures/Figure3_Adaptation_Accuracy.png")