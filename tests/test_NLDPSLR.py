from comparison.NLDPSLR import NLDPSLR
import numpy as np


def test_NLDPSLR():

    m = 20
    n = 800 * m
    d = 4
    s = 2
    X = np.random.normal(0, 1, size = (n, d))
    y = X[:, :s].sum(axis = 1)
    
  
    Xy = np.zeros(d)
    for i in range(n):
        Xy += X[i] * y[i]
    model = NLDPSLR(epsilon = 200,
                    r = 10,
                    tau1 = 5, 
                    tau2 = 5,
                    lamda = 0.1)
    model.fit(X, y)
    
    X_test = np.random.normal(0, 1, size = (n, d))
    y_test = X_test[:, :s].sum(axis = 1)
    mse = np.mean((y_test - model.predict(X_test)) ** 2)
    print(model.coef_)
    print(mse)