import pandas as pd
df = pd.read_csv('train.csv')
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns='Cabin', inplace=True)
df.drop_duplicates(inplace=True)
df.to_csv('cleaned_titanic.csv', index=False)
print("Data cleaned and saved to cleaned_titanic.csv ✅")