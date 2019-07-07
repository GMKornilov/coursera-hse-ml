import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train = pandas.read_csv("perceptron-train.csv", header=None)
test = pandas.read_csv("perceptron-test.csv", header=None)

y_train = train[0]
x_train = train.loc[:, 1:]


y_test = test[0]
x_test = test.loc[:, 1:]

clf = Perceptron(random_state=241)
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
score = accuracy_score(y_test, predictions)
print(score)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

clf = Perceptron(random_state=241)
clf.fit(x_train_scaled, y_train)
predictions_scaled = clf.predict(x_test_scaled)
score_scaled = accuracy_score(y_test, predictions_scaled)
print(score_scaled)
print(score_scaled - score)
with open("1", "w") as f:
    print(round(score_scaled - score, 3), file=f, end="")

