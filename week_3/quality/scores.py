import pandas
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve


def dict_mx(d):
    max_key = None
    mx = -1
    for i in d:
        if d[i] > mx:
            mx = d[i]
            max_key = i
    return max_key


data = pandas.read_csv("scores.csv")
y = data["true"]
columns = data.columns
roc = dict()
curve = dict()
for i in columns[1:]:
    x = data[i]
    score = roc_auc_score(y, x)
    c = precision_recall_curve(y, x)
    roc[i] = score
    df = pandas.DataFrame({"precision": c[0], "recall": c[1]})
    mx = df[df["recall"] >= 0.7]["precision"].max()
    curve[i] = mx

print(roc)
print(curve)

with open("3", "w") as f:
    print(dict_mx(roc), file=f, end="")
with open("4", "w") as f:
    print(dict_mx(curve), file=f, end="")