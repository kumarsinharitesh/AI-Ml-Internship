Iris Flower Classification using KNN

This project demonstrates how to use the K-Nearest Neighbors (KNN) algorithm to classify iris flowers into different species based on petal and sepal measurements. It includes steps like feature normalization, hyperparameter tuning, model evaluation, and decision boundary visualization.

📂 Dataset
We used the popular Iris dataset, which contains:

150 samples

4 features: Sepal Length, Sepal Width, Petal Length, Petal Width

3 classes: Setosa, Versicolor, Virginica

Each row represents an iris flower with its measurements and species label.

📌 Project Steps
✅ 1. Data Preprocessing
Removed the Id column.

Encoded species labels as numeric values.

Normalized the features using StandardScaler.

✅ 2. Model: K-Nearest Neighbors
Implemented KNN using sklearn.neighbors.KNeighborsClassifier.

Trained the model on a split dataset (80% train, 20% test).

Tested different values of K (1 to 10).

✅ 3. Evaluation
Computed accuracy for each value of K.

Chose the best K based on maximum accuracy.

Displayed confusion matrix for final model.

✅ 4. Decision Boundary Visualization
Used only the first 2 features to plot decision boundaries in 2D.

Visualized how the KNN model classifies regions in the feature space.

🔍 Results
Best K: Varies based on random state (e.g., K=3 or K=5 often perform best).

Accuracy: Usually between 95%–100% on test data.

KNN performs exceptionally well on the Iris dataset due to its simplicity and clean class separation.