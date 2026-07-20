import numpy as np
import matplotlib.pyplot as plt
import os

# Create figures folder
os.makedirs("figures", exist_ok=True)

models = {
    "Logistic Regression": "results/roc_data/logistic_regression_roc.npz",
    "Decision Tree": "results/roc_data/decision_tree_roc.npz",
    "Random Forest": "results/roc_data/random_forest_roc.npz",
    "SVM": "results/roc_data/svm_roc.npz",
    "Adaptive MLP": "results/roc_data/adaptive_mlp_roc.npz"
}

plt.figure(figsize=(10,8))

for model_name, file_path in models.items():

    data = np.load(file_path)

    fpr = data["fpr"]
    tpr = data["tpr"]

    auc = np.trapz(tpr, fpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC={auc:.3f})"
    )

# Random classifier
plt.plot([0, 1], [0, 1], "k--", linewidth=1)

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves of Disease Prediction Models", fontsize=14)

plt.legend(loc="lower right", fontsize=10)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "figures/Figure1_ROC_Curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nROC Curves saved to figures/roc_curves.png")