from ULDPFS import OLSRegressor, range_single, mean_single, ULDPMean, NULDPProtocol, GradientOLS, IULDPProtocol
import numpy as np
from scipy.stats import multivariate_normal


def test_single_range():
    x = np.random.rand(200) * 0.1 - 0.5
    print(x[:20])
    B = 1
    num_bins = 2**4
    epsilon = 0.3
    print(range_single(x, B, num_bins, epsilon))

def test_single_mean():
    x = np.random.rand(10) * 0.1 
    epsilon = 4
    
    print(mean_single(x, -0.1, 0.1, epsilon))



def test_uldp_mean():
    d = 4
    n = 1000
    epsilon = 1
    mean = np.random.choice(d, d) 
    X = multivariate_normal.rvs(mean = mean, size = n)

    est_mean = ULDPMean(epsilon = epsilon, 
                         num_bins = 2**7, 
                         B = 1).apply(X)
    
    print(mean)
    print(est_mean)
    

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

def test_nuldp_protocol():
    d = 500
    d1 = 10
    d2 = d - d1
    m = 50
    n = 100
    epsilon = 4
    
    X_list, y_list = [], []
    for i in range(n):
        X, y = generate(d, d1, d2, m)
        X_list.append(X)
        y_list.append(y)

    est_coef = NULDPProtocol([0,1,2,3,4],
                 epsilon,
                 num_bins = 2**3,
                 B = 1,
                 ).apply(X_list, y_list)
    

    
    print("nuldp", est_coef)
    
    
    
    
    
   

def test_OLS_regressor():
    
    d = 1000
    d1 = 30
    d2 = d - d1
    n = 30
    
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
    beta = OLSRegressor(feature_index = [i for i in range(10, 20)]).apply(X, y)
    print(beta)




def test_gradient_solver():

    d = 1000
    d1 = 30
    d2 = d - d1
    n = 300
    
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
    beta = GradientOLS(feature_index = [i for i in range(10, 20)]).apply(X, y, np.ones(10) * (1))
    print(beta)





def test_iuldp_protocol():
    d = 500
    d1 = 10
    d2 = d - d1
    m = 200
    n = 300
    epsilon = 4
    
    X_list, y_list = [], []
    for i in range(n):
        X, y = generate(d, d1, d2, m)
        X_list.append(X)
        y_list.append(y)

    est_coef = IULDPProtocol([0,1,2,3,4],
                 epsilon,
                 num_bins = 2**4,
                 B = 3,
                 batch_size = 10,
                 lr_const = 0.1,
                 lr_power = 0.2, 
                 ).apply(X_list, y_list)
    
    print("iuldp", est_coef)

    




