from comparison.LDPIHT import LDPIHT
import numpy as np


def test_LDPIHT():

    m = 200
    n = 800 * m
    d = 512
    s = 10
    X = np.random.normal(0, 1, size = (n, d))
    y = X[:, :s].sum(axis = 1)
    model = LDPIHT(epsilon = 1 / m, T = 10, s = s, eta = 0.1)
    model.fit(X, y)
    
    X_test = np.random.normal(0, 1, size = (n, d))
    y_test = X_test[:, :s].sum(axis = 1)
    mse = np.mean((y_test - model.predict(X_test)) ** 2)
    print(model.coef_)
    print(mse)