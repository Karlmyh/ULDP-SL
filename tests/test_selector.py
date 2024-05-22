

import numpy as np
from scipy.stats import multivariate_normal
from ULDPFS import ScreeningSelector, LassoSelector, PostLassoSelector


def test_screening_selector():
    
    d = 1000
    d1 = 30
    d2 = d - d1
    n = 80
    
    # correlated features
    mean = np.zeros(d1)
    cov = np.array([[2**(-np.abs(i - j)/2) for i in range(d1)] for j in range(d1)])
    p = np.zeros(d1)
    p[10:20] = 1
    X1 = multivariate_normal.rvs(mean = mean, cov = cov, size = n)
    y = np.dot(X1, p) + np.random.normal(size = n)

    # independent features
    X2 = multivariate_normal.rvs(size = (n, d2))
    X = np.concatenate((X1, X2), axis = 1)

    # selection
    clf = ScreeningSelector(num_features = 10).fit(X, y)
    print(clf.indexes)
    print(clf.get_idx())



def test_lasso_selector():
    
    d = 1000
    d1 = 30
    d2 = d - d1
    n = 80
    
    # correlated features
    mean = np.zeros(d1)
    cov = np.array([[2**(-np.abs(i - j)/2) for i in range(d1)] for j in range(d1)])
    p = np.zeros(d1)
    p[10:20] = 1
    X1 = multivariate_normal.rvs(mean = mean, cov = cov, size = n)
    y = np.dot(X1, p) + np.random.normal(size = n)

    # independent features
    X2 = multivariate_normal.rvs(size = (n, d2))
    X = np.concatenate((X1, X2), axis = 1)

    # selection
    clf = LassoSelector(num_features = 10).fit(X, y)
    print(clf.indexes)
    print(clf.get_idx())



def test_post_lasso_selector():
    
    d = 1000
    d1 = 30
    d2 = d - d1
    n = 80
    
    # correlated features
    mean = np.zeros(d1)
    cov = np.array([[2**(-np.abs(i - j)/2) for i in range(d1)] for j in range(d1)])
    p = np.zeros(d1)
    p[10:20] = 1
    X1 = multivariate_normal.rvs(mean = mean, cov = cov, size = n)
    y = np.dot(X1, p) + np.random.normal(size = n)

    # independent features
    X2 = multivariate_normal.rvs(size = (n, d2))
    X = np.concatenate((X1, X2), axis = 1)

    # selection
    clf = PostLassoSelector(num_features = 10,
                            screening_num_features = 20).fit(X, y)
    print(clf.indexes)
    print(clf.get_idx())






    



    
