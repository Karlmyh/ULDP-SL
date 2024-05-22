from ULDPFS import ULDPFS, ULDPFS_IA
import numpy as np
from scipy.stats import multivariate_normal

def generate(d, d1, d2, m):
     # correlated features
    mean = np.zeros(d1)
    cov = np.array([[2**(-np.abs(i - j)/0.1) for i in range(d1)] for j in range(d1)])
    p = np.zeros(d1)
    p[:5] = 0.2
    X1 = multivariate_normal.rvs(mean = mean, cov = cov, size = m)
    y = np.dot(X1, p) + np.random.normal(size = m)

    # independent features
    X2 = multivariate_normal.rvs(size = (m, d2))
    X = np.concatenate((X1, X2), axis = 1)
    return X, y



def test_ULDPFS():
    
    d = 1000
    d1 = 10
    d2 = d - d1
    m = 200
    n = 400
    
    X_list, y_list = [], []
    for i in range(n):
        X, y = generate(d, d1, d2, m)
        X_list.append(X)
        y_list.append(y)

    params = {"epsilon": 4,
              "selector": "postlasso",
              "selector_params": {"num_features" : 5 // 2,
                                    "screening_num_features" : 5**2 // 4},
              "heavyhitter_params": {"min_hitters": 5,
                                       "alpha" : 0.1},
              "regressor_params": {"num_bins" : 2**4, "B" : 1}}
    
    clf = ULDPFS(**params).fit(X_list, y_list)
    print(clf.selected_indexes)
    print(clf.coef_)



def test_ULDPFS_IA():
    
    d = 1000
    d1 = 10
    d2 = d - d1
    m = 200
    n = 400
    
    X_list, y_list = [], []
    for i in range(n):
        X, y = generate(d, d1, d2, m)
        X_list.append(X)
        y_list.append(y)

    params = {"epsilon": 4,
              "selector": "postlasso",
              "selector_params": {"num_features" : 5 // 2,
                                    "screening_num_features" : 5**2 // 4},
              "heavyhitter_params": {"min_hitters": 5,
                                       "alpha" : 0.1},
              "regressor_params": {"num_bins" : 2**4, "B" : 3, "batch_size": 10, "lr_const": 0.1, "lr_power":0.2}}
    
    clf = ULDPFS_IA(**params).fit(X_list, y_list)
    print(clf.selected_indexes)
    print(clf.coef_)
