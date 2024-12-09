import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier

# Step 1: Generate Data
X_A, y_A = make_blobs(n_samples=100, centers=[[2.0, 2.0]], cluster_std=0.75, random_state=42)
X_B, y_B = make_blobs(n_samples=100, centers=[[3.0, 3.0]], cluster_std=0.75, random_state=42)
y_A[:] = 0  # Label for Class A
y_B[:] = 1  # Label for Class B

# Combine datasets
X = np.vstack((X_A, X_B))
y = np.hstack((y_A, y_B))

# Step 2: Train Neural Network
clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, activation='identity', solver='lbfgs', random_state=42)
clf.fit(X, y)

# Step 3: Plot Decision Boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict on mesh grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdBu, marker='o')
plt.title("Decision Plane (Diagonal Line)")
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.legend(['Class 1', 'Class 2'], loc="upper left")
plt.show()


