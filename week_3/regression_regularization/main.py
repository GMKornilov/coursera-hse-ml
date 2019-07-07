import pandas
import numpy as np
import gradient
from sklearn.metrics import roc_auc_score, accuracy_score

data = pandas.read_csv("data-logistic.csv", header=None)
X = data.loc[:, 1:]
y = data[0]

w_0 = gradient.fit(X, y)
w_1 = gradient.fit(X, y, C=10)

a_0 = gradient.a(X, *w_0)
a_1 = gradient.a(X, *w_1)

auc_0 = round(roc_auc_score(y, a_0), 3)
auc_1 = round(roc_auc_score(y, a_1), 3)

X = np.array(X)
y_0 = gradient.predict(X, w_0)
y_1 = gradient.predict(X, w_1)

print(w_0)
print(w_1)
print(accuracy_score(y, y_0), accuracy_score(y, y_1))


