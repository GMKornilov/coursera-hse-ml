from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
import numpy
import pandas


def accuracy(data, target, fold, p):
    est = KNeighborsRegressor(n_neighbors=5, weights="distance", p=p, metric="minkowski")
    score = cross_val_score(est, data, target, cv=fold, scoring="neg_mean_squared_error")
    return score.mean()


data = load_boston()
X = data["data"]
y = data["target"]
X = scale(X)

fold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for i in numpy.linspace(1, 10, 200):
    score = accuracy(X, y, fold, i)
    scores.append([i, score])
scores.sort(reverse=True, key=lambda x: x[1])
with open("1", "w") as f:
    print(scores[0][0], file=f, end="")
print(scores)
