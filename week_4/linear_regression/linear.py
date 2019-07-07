import pandas
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def transform_text(X):
    return X.str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)


train = pandas.read_csv("salary-train.csv")
test = pandas.read_csv("salary-test-mini.csv")

vec = TfidfVectorizer(min_df=5)
train_X_text = vec.fit_transform(transform_text(train["FullDescription"]))

train["LocationNormalized"].fillna('nan', inplace=True)
train["ContractTime"].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([train_X_text, X_train_categ])
y_train = train["SalaryNormalized"]

clf = Ridge(alpha=1, random_state=241)
clf.fit(X_train, y_train)

test_X_text = vec.transform(transform_text(test["FullDescription"]))
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_test = hstack([test_X_text, X_test_categ])
y_test = clf.predict(X_test)

y_test[0] = round(y_test[0], 2)
y_test[1] = round(y_test[1], 2)

with open("1", "w") as f:
    print(*y_test, end="", file=f)
