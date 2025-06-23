import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# lets get import the dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Basic dataset overview
print(df.info())
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# Encode categorical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Standardize numerical columns
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Outlier removal using IQR method
sns.boxplot(x='Fare', data=df)
plt.title('Fare Distribution with Outliers')
plt.show()

q1 = df['Fare'].quantile(0.25)
q3 = df['Fare'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

# Save cleaned data
df.to_csv('cleaned_titanic dataset.csv', index=False)
