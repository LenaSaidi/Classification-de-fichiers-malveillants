import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


df = pd.read_csv("ia_test_dataset.csv") 

print(df.shape)
print((df["target"].value_counts())*100/df.shape[0])

missing_values = df.isna().sum().sort_values(ascending=False)
print(missing_values)

df = df.dropna()

X = df.drop(columns=["target"], axis=1).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(X_train, "X_train.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(scaler, "scaler.pkl")

print(df.isna().sum().sort_values(ascending=False))
print(df.shape)
print((df["target"].value_counts())*100/df.shape[0])
print("Prétraitement terminé")