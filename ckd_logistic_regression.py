from ckd_preprocessing import load_ckd_data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = load_ckd_data()

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

prob = model.predict_proba(X_test)[:, 1]

print("="*50)
print("Logistic Regression")
print("="*50)

print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1 Score :", f1_score(y_test, pred))
print("ROC AUC  :", roc_auc_score(y_test, prob))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred))