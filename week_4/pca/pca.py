from sklearn.decomposition.pca import PCA
import numpy as np
import pandas as pd

prices = pd.read_csv("close_prices.csv")
X = prices.loc[:, "AXP":]
est = PCA(n_components=10)
est.fit(X)

total = 0.0
for num, val in enumerate(est.explained_variance_ratio_):
    total += val
    if total >= 0.9:
        with open("1", "w") as f:
            print(num + 1, file=f, end="")
            break

X0 = pd.DataFrame(est.transform(X))[0]

index = pd.read_csv("djia_index.csv")
corr = np.corrcoef(X0, index["^DJI"])[1][0]
with open("2", "w") as f:
    print(round(corr, 2), file=f, end="")

mx_company = X.columns[np.argmax(est.components_[0])]
with open("3", "w") as f:
    print(mx_company, file=f, end="")
print(mx_company)
