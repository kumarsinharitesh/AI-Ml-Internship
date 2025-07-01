import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# 1. Load and preprocess the data
df = pd.read_csv("Iris.csv")  # Make sure the file is in the same directory

# Drop ID column if exists
df.drop("Id", axis=1, inplace=True)

# Encode target labels
df["Species"] = df["Species"].astype("category").cat.codes

# Split into features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2 & 3. Try different values of K
k_values = range(1, 11)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k} -> Accuracy: {acc:.2f}")

# 4. Confusion Matrix for best K
best_k = k_values[np.argmax(accuracies)]
print(f"\nBest K = {best_k} with accuracy = {max(accuracies):.2f}")

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix (Best K)")
plt.show()

# 5. Visualize decision boundaries (only using 2 features for 2D plot)
from matplotlib.colors import ListedColormap

X_2d = X_scaled[:, :2]  # take first two features
model_2d = KNeighborsClassifier(n_neighbors=best_k)
model_2d.fit(X_2d, y)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ["red", "green", "blue"]

plt.contourf(xx, yy, Z, cmap=cmap_light)
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y, palette=cmap_bold, edgecolor="k")
plt.title(f"Decision Boundary (K={best_k}) using first 2 features")
plt.xlabel("Feature 1 (normalized)")
plt.ylabel("Feature 2 (normalized)")
plt.show()
