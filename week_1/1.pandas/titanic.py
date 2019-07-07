import pandas as pd
import numpy

data = pd.read_csv("titanic.csv", index_col="PassengerId")

sex = data["Sex"].value_counts()
with open("1", "w") as f:
    print(sex["male"], sex["female"], file=f, end="")

alive = data["Survived"].value_counts()
survived = alive[1]
died = alive[0]
total = survived + died
with open("2", "w") as f:
    print(round(survived * 100 / total, 2), file=f, end="")

total = 0
classes = data["Pclass"].value_counts()
for i in classes:
    total += i
first_class = classes[1]
with open("3", "w") as f:
    print(round(first_class * 100 / total, 2), file=f, end="")
    
ages = data["Age"]
with open("4", "w") as f:
    print(ages.mean(), ages.median(), file=f, end="")

corr = data["SibSp"].corr(data["Parch"], method="pearson")
with open("5", "w") as f:
    print(corr, file=f, end="")
