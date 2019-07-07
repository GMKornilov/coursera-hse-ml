import pandas as pd
import numpy
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("titanic.csv")

data = data[["Pclass", "Sex", "Fare", "Age", "Survived"]]

data.dropna(axis=0, how="any", inplace=True)

data["Sex"].replace(["female", "male"], [0, 1], inplace=True)


x = data[["Pclass", "Sex", "Fare", "Age"]]
y = data["Survived"]


clf = DecisionTreeClassifier(random_state=241)
clf.fit(x, y)

importances = clf.feature_importances_
features = x.columns.values

importances = list(zip(importances, features))
importances.sort(reverse=True, key=lambda a: a[0])
importances = importances[:2]

with open("ans", "w") as f:
    for i in importances:
        print(i[1], file=f, end=" ")

print(importances)
