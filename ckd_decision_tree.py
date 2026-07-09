from ckd_preprocessing import load_ckd_data

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# Load data
X_train, X_test, y_train, y_test = load_ckd_data()

# Create model
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# Results
print("=" * 50)
print("Decision Tree")
print("=" * 50)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))