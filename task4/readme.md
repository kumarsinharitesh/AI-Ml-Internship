# Logistic Regression on Breast Cancer Dataset

## 📁 Dataset

- The dataset used is `data.csv` (Breast Cancer Wisconsin Diagnostic Data).
- It contains various measurements of cell nuclei in digitized images of breast masses.

---

## 🧠 Problem Statement

Predict whether a tumor is **Malignant (1)** or **Benign (0)** using logistic regression, based on features like:
- Radius, texture, perimeter, area
- Smoothness, compactness, concavity
- And their worst/mean values



## ✅ Steps Covered in Code

1. **Load and clean the dataset**  
   - Drop unnecessary columns  
   - Encode the target variable (`diagnosis`)  

2. **Preprocess the data**  
   - Split into train and test sets  
   - Standardize the features  

3. **Train the Model**  
   - Fit a Logistic Regression model on the training set  

4. **Evaluate the Model**  
   - Confusion Matrix  
   - Classification Report (Precision, Recall, F1-score)  
   - ROC Curve & AUC Score  

5. **Explain the Sigmoid Function**  
   - Probability prediction is based on the sigmoid:  
     \[
     \sigma(z) = \frac{1}{1 + e^{-z}}
     \]


## 📦 Requirements

Install dependencies using:

bash
pip install pandas numpy matplotlib seaborn scikit-learn
