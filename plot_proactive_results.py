import pandas as pd
import matplotlib.pyplot as plt
import os

# Create figures folder
os.makedirs("figures", exist_ok=True)

# Load results
df = pd.read_csv("results/proactive_results.csv")

metrics = ["Accuracy", "F1 Score"]
values = [df.loc[0, "Accuracy"], df.loc[0, "F1 Score"]]

plt.figure(figsize=(6,6))

plt.bar(metrics, values)

plt.ylim(0, 1)

plt.ylabel("Score")
plt.title("Proactive Detection Performance")

# Add values on top of bars
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

plt.tight_layout()

plt.savefig(
    "figures/Figure5_Proactive_Detection.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to figures/Figure5_Proactive_Detection.png")