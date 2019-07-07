from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np

newsgroups = datasets.fetch_20newsgroups(
    subset="all",
    categories=["alt.atheism", "sci.space"]
)
texts = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

grid = {"C": np.power(10.0, np.arange(-5, 6))}
cv = KFold(random_state=241, n_splits=5, shuffle=True)
clf = SVC(kernel="linear", random_state=241)
gs = GridSearchCV(clf, grid, scoring="accuracy", cv=cv)
gs.fit(X, y)

best = gs.best_estimator_
best.fit(X, y)
print(best.coef_)
res = list(zip(best.coef_.data, best.coef_.indices))
res.sort(reverse=True, key=lambda x: abs(x[0]))
res = res[:10]
top_words = []
for i in res:
    top_words.append(vectorizer.get_feature_names()[i[1]])
top_words.sort()
print(*top_words)
with open("ans", "w") as f:
    print(*top_words, file=f, end="")
