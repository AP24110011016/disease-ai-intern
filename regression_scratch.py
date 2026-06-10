import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

# Parameters
m = 0
b = 0
learning_rate = 0.01
epochs = 1000

n = len(X)
cost_history = []

# Gradient Descent
for epoch in range(epochs):
    y_pred = m * X + b

    cost = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)
    cost_history.append(cost)

    dm = (1 / n) * np.sum((y_pred - y) * X)
    db = (1 / n) * np.sum(y_pred - y)

    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

# Plot Cost vs Epochs
plt.plot(range(epochs), cost_history)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost vs Epochs")
plt.show()