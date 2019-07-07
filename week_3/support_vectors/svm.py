import pandas
from sklearn.svm import SVC

data = pandas.read_csv("svm-data.csv", header=None)
X = data.loc[:, 1:]
y = data[0]

svm = SVC(kernel="linear", C=100000, random_state=241)
svm.fit(X, y)
a = svm.support_
with open("ans", "w") as f:
    for i in range(len(a) - 1):
        print(a[i] + 1, file=f, end=" ")
    print(a[-1] + 1, file=f, end="")
