import numpy as np


def calc_w1(X, y, w1, w2, k, C):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[1][i] * (1.0 - 1.0 / (1.0 + np.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i]))))
    return w1 + (k * (1.0 / l) * S) - k * C * w1


def calc_w2(X, y, w1, w2, k, C):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[2][i] * (1.0 - 1.0 / (1.0 + np.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i]))))
    return w2 + (k * (1.0 / l) * S) - k * C * w2


def fit(X, y, k=0.1, steps=10000, C=0):
    w1 = 0.0
    w2 = 0.0
    ll = 0
    for i in range(steps):
        last_w1, last_w2 = w1, w2
        w1, w2 = calc_w1(X, y, w1, w2, k, C), calc_w2(X, y, w1, w2, k, C)
        ll += 1
        dist = np.sqrt(
            (w1 - last_w1) ** 2 + (w2 - last_w2) ** 2
        )
        if dist <= 1e-5:
            break
    print(ll)
    return [w1, w2]


def a(X, w1, w2):
    return 1.0 / (1 + np.exp(-w1 * X[1] - w2 * X[2]))


def predict(X, w):
    res = [None] * len(X)
    for i in range(len(res)):
        now = 0
        for j in range(len(X[i])):
            now += X[i][j] * w[j]
        if now >= 0:
            res[i] = 1
        else:
            res[i] = -1
    return res
