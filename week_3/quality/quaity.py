import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

data_1 = pandas.read_csv("classification.csv")
a = [[0, 0], [0, 0]]
y = data_1["true"]
pred = data_1["pred"]
for i in range(len(y)):
    a[pred[i]][y[i]] += 1
tp = a[1][1]
fp = a[1][0]
fn = a[0][1]
tn = a[0][0]

with open("1", "w") as f:
    print(tp, fp, fn, tn, file=f, end="")

acc = accuracy_score(y, pred)
prec = precision_score(y, pred)
recall = recall_score(y, pred)
f_score = f1_score(y, pred)

acc = round(acc, 2)
prec = round(prec, 2)
recall = round(recall, 2)
f_score = round(f_score, 2)

with open("2", "w") as f:
    print(acc, prec, recall, f_score, file=f, end="")

