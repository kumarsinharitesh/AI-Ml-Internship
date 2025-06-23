# Titanic Dataset Preprocessing

This project focuses on cleaning and preparing the Titanic dataset for further data analysis or machine learning modeling. The dataset contains passenger information, and the goal is to preprocess the data by handling missing values, encoding categorical variables, removing outliers, and scaling numerical features.

#📁 Dataset

The dataset used is the **Titanic-Dataset.csv**, which includes features such as:

- PassengerId
- Survived
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

#🔧 Steps Performed

### 1. Importing Required Libraries
Used libraries:  
- `pandas` for data manipulation  
- `seaborn` and `matplotlib.pyplot` for visualization  
- `sklearn.preprocessing.StandardScaler` for scaling numerical features

### 2. Loading and Exploring the Data
- Read the CSV file using `pandas`.
- Explored the data using `.info()`, `.head()`, `.describe()`, and `.isnull().sum()`.

### 3. Handling Missing Values
- Filled missing values in `Age` using the median.
- Filled missing `Embarked` values using the mode.
- Dropped the `Cabin` column due to excessive missing values.

### 4. Encoding Categorical Features
- Converted categorical columns (`Sex`, `Embarked`) to numeric using one-hot encoding (`pd.get_dummies`).

### 5. Feature Scaling
- Scaled `Age` and `Fare` columns using `StandardScaler`.

### 6. Outlier Detection and Removal
- Used the IQR (Interquartile Range) method to identify and remove outliers from the `Fare` column.
- Visualized outliers using a boxplot.

### 7. Saving Cleaned Data
- The cleaned dataset is saved as `cleaned_titanic.csv`.

#📊 Output

- `cleaned_titanic.csv` – A clean, numeric, and scaled version of the original Titanic dataset ready for machine learning tasks.

# ✅ Requirements

Install the required libraries using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
(make sure that you install the required libraries )

