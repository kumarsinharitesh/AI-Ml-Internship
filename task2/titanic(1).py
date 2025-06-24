import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("Titanic-Dataset.csv")

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

num_cols = ['Age', 'Fare', 'SibSp', 'Parch']

plt.figure(figsize=(10, 6))
df[num_cols].hist(bins=20, layout=(2, 2), color='skyblue')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='Survived', y=col, data=df)
    plt.title(f'{col} vs Survived')
plt.tight_layout()
plt.show()

sns.pairplot(df[['Survived', 'Age', 'Fare', 'SibSp', 'Parch']], hue='Survived', diag_kind='hist')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

fig = px.histogram(df, x="Pclass", color="Survived", barmode="group", facet_col="Sex",
                   title="Survival by Class and Gender", category_orders={"Pclass": [1, 2, 3]})
fig.show()
