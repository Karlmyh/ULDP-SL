'''
Used for first phase variable selection.
Each object is able to select variable based on local samples and reture a index of selected variable.
'''

import numpy as np
import math
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from scipy.stats import norm
from scipy.linalg import hadamard





class OLSRegressor(object):
    def __init__(self, 
                 feature_index):
        '''
        Input:
        ---------
        feature_index, the index of the selected feature
        '''
        
        self.feature_index = feature_index
        
    def apply(self, X, y):
        '''
        Input:
        ---------
        X, the local samples
        y, the local labels

        Output:
        ---------
        self, the fitted object
        '''
        
        reduced_X = X[:, self.feature_index].reshape(-1, len(self.feature_index))
        model = LinearRegression(fit_intercept = False).fit(reduced_X, y)
        return model.coef_
    
    
    
class GradientOLS(object):
    def __init__(self, 
                 feature_index):
        '''
        Input:
        ---------
        feature_index, the index of the selected feature
        '''
        
        self.feature_index = feature_index
        
    def apply(self, X, y, beta):
        '''
        Input:
        ---------
        X, the local samples
        y, the local labels
        beta, the current parameter

        Output:
        ---------
        self, the fitted object
        '''
        reduced_X = X[:, self.feature_index]
        # averaged gradient of OLS
        gradient = reduced_X.T @ (reduced_X @ beta - y) * 2 / len(y)
        return gradient.ravel()
    
    
def ceil_to_power_2(x):
    if x == 0:
        return 1
    else:
        return int(2**math.ceil(math.log(x, 2)))


def range_single(x, B, num_bins, epsilon):
    # num_bins is power of 2
    # x is a vector of length n
    # partition the range [-B, B] into len-tau bins

    x = [element for element in x if np.abs(element) <= B]
    hadamard_matrix = hadamard(num_bins)
    edges = np.linspace(- B, B, num_bins + 1)
    bin_indexes = np.digitize(x, edges) - 1
    aepsilon = (1 + math.exp( - epsilon)) / (1 - math.exp( - epsilon))

    freq_vec = np.zeros(num_bins)
    for bin_index in bin_indexes:
        j = np.random.choice(num_bins)
        Hj = hadamard_matrix[j]  * aepsilon
        prob = Hj[bin_index] / aepsilon**2 / 2 + 0.5
        freq_vec += Hj * np.random.choice([-1, 1 ], p = [1 - prob, prob ])

    # return 3 tau
    frequentest_index = freq_vec.argmax()
    inverval_l, inverval_r = edges[frequentest_index], edges[frequentest_index + 1]
    inverval_l = inverval_l - (inverval_r - inverval_l) / 2
    inverval_r = inverval_r + (inverval_r - inverval_l) / 2
    return inverval_l, inverval_r


def mean_single(x, inverval_l, inverval_r, epsilon):

    noisy_x = np.array([x + np.abs(inverval_r - inverval_l) / epsilon * np.random.laplace() for x in x])
    return noisy_x.mean()

    


class ULDPMean(object):
    def __init__(self, 
                 epsilon,
                 num_bins = 2**3,
                 B = 1,
                 ):
        '''
        Estimate the final regression coefficient based on the locally coefficients and Gaussian debias.
        Input:
        ---------
        epsilon, the privacy budget
        
        '''
        
        self.epsilon = epsilon
        self.num_bins = num_bins
        self.B = B
        
        
    def apply(self, Beta):

        self.n, self.s = Beta.shape

        self.K = ceil_to_power_2(self.s)

        # fill beta with 0 to K dim
        Beta = np.concatenate((Beta, np.zeros((self.n, self.K - self.s))), axis = 1)
        # rotation
        U = hadamard(self.K) * np.random.choice([-1, 1], self.K) / math.sqrt(self.K)
        rotated_Beta = Beta @ U 

        mean_final = np.zeros(self.K)
        for l in range(self.K):
            # range
            inverval_l, interval_r = range_single(rotated_Beta[: (self.n // 2) , l], self.B, self.num_bins, self.epsilon / self.K)
            # mean
            mean_final[l] = mean_single(rotated_Beta[int(self.n // self.K * l):int(self.n // self.K * (l + 1)), l], inverval_l, interval_r, self.epsilon)
            

        return (mean_final @ U.T)[:self.s]
    



class NULDPProtocol(object):
    def __init__(self, 
                 feature_index,
                 epsilon,
                 num_bins = 2**3,
                 B = 1,
                 ):
        '''
        Estimate the final regression coefficient based on the local samples using noninteractive protocol.
        Input:
        ---------
        feature_index, the index of the selected feature
        epsilon, the privacy budget
        num_bins, the number of bins in mean estimation
        B, the upper bound of the range
        '''

        self.mean_estimator = ULDPMean(epsilon = epsilon, 
                                       num_bins = num_bins, 
                                       B = B)
        self.local_estimator = OLSRegressor(feature_index = feature_index)
        

    def apply(self, X_list, y_list):
        '''
        Input:
        ---------
        X_list, the local samples
        y_list, the local labels

        Output:
        ---------
        self, the fitted object
        '''
        local_coeffs = [self.local_estimator.apply(X, y) for X, y in zip(X_list, y_list)]
        self.mean_coef = self.mean_estimator.apply(np.array(local_coeffs))
        return self.mean_coef

        

class IULDPProtocol(object):
    def __init__(self, 
                 feature_index,
                 epsilon,
                 num_bins = 2**3,
                 B = 1,
                 batch_size = 3,
                 lr_const = 1,
                 lr_power = 1
                 ):
        '''
        Estimate the final regression coefficient based on the local samples using interactive protocol.
        Input:
        ---------
        feature_index, the index of the selected feature
        epsilon, the privacy budget
        num_bins, the number of bins in mean estimation
        B, the upper bound of the rangeo
        batch_size, the number of user to consider each descent
        lr_const, the learning rate
        '''
        self.s = len(feature_index)
        self.mean_estimator = ULDPMean(epsilon = epsilon, 
                                       num_bins = num_bins, 
                                       B = B)
        self.gradient_estimator = GradientOLS(feature_index = feature_index)
        self.batch_size = batch_size
        self.lr_const = lr_const
        self.lr_power = lr_power
        

    def apply(self, X_list, y_list):
        '''
        Input:
        ---------
        X_list, the local samples
        y_list, the local labels

        Output:
        ---------
        self, the fitted object
        '''
        self.n = len(X_list)
        self.m ,self.d = X_list[0].shape
        
        theta = np.zeros(self.s)
        thetaag = theta.copy()
        
        for iter in range(self.n // self.batch_size):
            gamma = (iter + 2) / 2
            eta = self.lr_const * (iter + 1)** self.lr_power
            thetamd = theta * gamma**(-1) + thetaag * (1 - gamma**(-1))
            # compute the gradient
            gradients = np.array([self.gradient_estimator.apply(X, y, thetamd) 
                                  for X, y in zip(X_list[iter * self.batch_size : (iter+1) * self.batch_size], y_list[iter * self.batch_size : (iter+1) * self.batch_size])])
            mean_gradient = self.mean_estimator.apply(gradients)
            # update
            theta = thetamd - eta * mean_gradient
            thetaag = theta * gamma**(-1) + thetaag * (1 - gamma**(-1))
        
        
        self.mean_coef = thetaag
        return self.mean_coef
    
    


    




        
       