# Titanic EDA Project
# ===================

# 📦 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
sns.set(style="whitegrid")
plt.style.use('ggplot')

# 📥 2. Load Dataset
df = pd.read_csv('train.csv')
print("Dataset Loaded. Shape:", df.shape)

# 🧾 3. Overview of Data
print(df.info())
print(df.describe(include='all'))

# 🔎 4. Check Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# 🧹 5. Data Cleaning
# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin due to too many missing values
df.drop(columns='Cabin', inplace=True)

# Drop duplicates if any
df.drop_duplicates(inplace=True)

# 🧪 6. Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 🔍 7. Exploratory Data Analysis (EDA)

# 7.1 Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()

# 7.2 Pclass vs Survival
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# 7.3 Sex vs Survival
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# 7.4 Age Distribution
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

# 7.5 Age vs Survival (Boxplot)
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()

# 7.6 Family Size vs Survival
sns.countplot(x='FamilySize', hue='Survived', data=df)
plt.title("Family Size vs Survival")
plt.show()

# 7.7 Embarked vs Survival
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title("Survival by Embarked Port")
plt.show()

# 7.8 Fare Distribution
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

# 7.9 Correlation Matrix
plt.figure(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# ✅ 8. Conclusion
print("\n🧠 Key Findings:")
print("- Women and 1st Class passengers had higher survival chances.")
print("- Age, Fare, and Family Size moderately influenced survival.")
print("- Cabin was dropped due to many missing values.")
print("- Embarked and Pclass showed some survival differences.")
