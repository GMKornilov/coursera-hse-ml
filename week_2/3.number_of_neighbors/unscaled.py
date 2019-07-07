import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
import numpy


def accuracy(fold, X, y):
    scores = []
    for i in range(1, 51):
        classifier = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(classifier, X, y, cv=fold, scoring="accuracy")
        mean = numpy.array(score).mean()
        scores.append([i, mean])
    return scores


data = pandas.read_csv("wine.data", header=None)

X = data.loc[:, 1:]
y = data[0]

fold = KFold(n_splits=5, shuffle=True, random_state=42)

res_unscaled = accuracy(fold, X, y)
res_unscaled.sort(reverse=True, key=lambda x:x[1])
unscaled_num_of_neighbors = res_unscaled[0][0]
unscaled_score = round(res_unscaled[0][1], 2)

with open("1", "w") as f:
    print(unscaled_num_of_neighbors, file=f, end="")
with open("2", "w") as f:
    print(unscaled_score, file=f, end="")


X = scale(X)
res_scaled = accuracy(fold, X, y)
res_scaled.sort(reverse=True, key=lambda x:x[1])
scaled_num_of_neighbors = res_scaled[0][0]
scaled_score = round(res_scaled[0][1], 2)

with open("3", "w") as f:
    print(scaled_num_of_neighbors, file=f, end="")
with open("4", "w") as f:
    print(scaled_score, file=f, end="")
