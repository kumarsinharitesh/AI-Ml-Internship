🫀 Heart Disease Prediction using Decision Trees and Random Forest
This project focuses on building predictive models to identify heart disease using a medical dataset. We train a Decision Tree Classifier and a Random Forest Classifier, visualize the tree, analyze overfitting, and evaluate the models using cross-validation. We also interpret the most important features that influence predictions.

📁 Dataset
The dataset used is [heart.csv] containing patient information like age, blood pressure, cholesterol levels, and more. The target variable indicates the presence (1) or absence (0) of heart disease.

🔧 Technologies Used
Python 3

Pandas

Scikit-learn

Matplotlib

Seaborn

🧠 Steps Performed
1. Train Decision Tree & Visualize
Trained a decision tree with a controlled depth (max_depth=4).

Visualized the decision-making flow using plot_tree().

2. Analyze Overfitting
Compared performance with limited tree depth to avoid overfitting.

Used cross-validation to verify generalization.

3. Train Random Forest
Trained a Random Forest with 100 trees.

Achieved high accuracy due to ensemble learning.

4. Feature Importance
Analyzed which features contribute most using feature_importances_.

Plotted top features like chest pain type, thalach, oldpeak.

5. Model Evaluation
Compared accuracy scores and classification reports.

Used 5-fold cross-validation for robust evaluation.

📊 Results Summary
Model	Test Accuracy	Cross-Validation Accuracy
Decision Tree	~80%	~83%
Random Forest	~98%	~99.7%

Random Forest performed significantly better, showing that ensemble methods are more robust and accurate in medical predictions.

📈 Key Insights
Chest pain type (cp), number of major vessels (ca), and max heart rate (thalach) are among the most important features.

Random Forest is highly accurate and avoids overfitting better than a single decision tree.

✅ How to Run
pip install pandas scikit-learn matplotlib seaborn
python your_script_name.py
Make sure heart.csv is in the same directory as your script.

📌 Future Improvements
*aAdd hyperparameter tuning (GridSearchCV).

*Use SHAP or LIME for advanced feature explanation.

*Deploy model via Flask or Streamlit for interactive use.