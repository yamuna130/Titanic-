import pandas as pd
df = pd.read_csv('cleaned_titanic.csv')
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df.to_csv('final_titanic.csv', index=False)
print("✅ Feature engineering complete. Saved to final_titanic.csv")