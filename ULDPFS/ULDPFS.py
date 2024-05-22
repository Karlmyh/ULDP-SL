
'''
The main user level locally differentially private feature selection class.
'''

import numpy as np
import math
from collections import Counter
from .heavy_hitters import HeavyHitters
from .regressor import NULDPProtocol, IULDPProtocol
from .selector import LassoSelector, ScreeningSelector, PostLassoSelector


SELECTOR = {"lasso": LassoSelector,
            "screening": ScreeningSelector,
            "postlasso": PostLassoSelector}





class ULDPFS(object):
    '''
    The user level locally differentially private feature selection class.
    '''


    def __init__(self, 
                 epsilon = 10,
                 selector = "screening",
                 selector_params = {"num_features" : 2,
                                    "screening_num_features" : 10},
                 heavyhitter_params = {"min_hitters": 10,
                                       "alpha" : 0.1},
                 regressor_params = {"num_bins" : 2**2, "B" : 1},
                 ):
        '''
        Input:
        ---------
        epsilon, the privacy budget
        selector, the selector used to select the features
        selector_params, the parameters for selector
        regressor_params, the parameters for regressor
        '''
        self.epsilon = epsilon
        self.selector = selector
        self.selector_params = selector_params
        self.heavyhitter_params = heavyhitter_params
        self.regressor_params = regressor_params

    def fit(self, X_list, y_list):
        '''
        Fit the model on the list of inputs of users.

        Parameters
        ----------
        X_list : length n list.
            The list of local samples.
        y_list : length n list.
            The list of local labels.

        Returns
        -------
        self : object
            Returns self.
        '''

        self.n = len(X_list)
        self.n1 = self.n // 2
        self.n2 = self.n - self.n1 
        self.m_list = [len(X) for X in X_list]
        self.d = X_list[0].shape[1]
        self.X_list = X_list
        self.y_list = y_list

        # compute the global feature set
        indexes = [SELECTOR[self.selector](**self.selector_params).get_idx(X, y) for X, y in zip(X_list[:self.n1], y_list[:self.n1])]

       
        # get the heavy hitters
        self.selected_indexes, _ = HeavyHitters(epsilon = self.epsilon, 
                                                d = self.d, 
                                                user_values = indexes,
                                                **self.heavyhitter_params).apply()
        
        self.low_dim_coef = NULDPProtocol(feature_index = self.selected_indexes,
                      epsilon = self.epsilon,
                      **self.regressor_params
                    ).apply(X_list[self.n1:], y_list[self.n1:])
        
        self.coef_ = np.zeros(self.d)
        self.coef_[self.selected_indexes] = self.low_dim_coef
        
        return self
    
    def predict(self, X):
        '''
        Predict the label of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted labels.
        '''
        
        X_selected = X[:, self.selected_indexes]
        
        return np.dot(X_selected, self.low_dim_coef).ravel()




class ULDPFS_IA(object):
    '''
    The user level locally differentially private feature selection class with interactive optimization.
    '''


    def __init__(self, 
                 epsilon = 10,
                 selector = "screening",
                 selector_params = {"num_features" : 2,
                                    "screening_num_features" : 10},
                 heavyhitter_params = {"min_hitters": 10,
                                       "alpha" : 0.1},
                 regressor_params = {"num_bins" : 2**2, "B" : 1, "batch_size": 3, "lr_const": 0.1, "lr_power":0.1},
                 ):
        '''
        Input:
        ---------
        epsilon, the privacy budget
        selector, the selector used to select the features
        selector_params, the parameters for selector
        regressor_params, the parameters for regressor
        '''
        self.epsilon = epsilon
        self.selector = selector
        self.selector_params = selector_params
        self.heavyhitter_params = heavyhitter_params
        self.regressor_params = regressor_params

    def fit(self, X_list, y_list):
        '''
        Fit the model on the list of inputs of users.

        Parameters
        ----------
        X_list : length n list.
            The list of local samples.
        y_list : length n list.
            The list of local labels.

        Returns
        -------
        self : object
            Returns self.
        '''

        self.n = len(X_list)
        self.n1 = self.n // 2
        self.n2 = self.n - self.n1 
        self.m_list = [len(X) for X in X_list]
        self.d = X_list[0].shape[1]
        self.X_list = X_list
        self.y_list = y_list

        # compute the global feature set
        indexes = [SELECTOR[self.selector](**self.selector_params).get_idx(X, y) for X, y in zip(X_list[:self.n1], y_list[:self.n1])]

       
        # get the heavy hitters
        self.selected_indexes, _ = HeavyHitters(epsilon = self.epsilon, 
                                                d = self.d, 
                                                user_values = indexes,
                                                **self.heavyhitter_params).apply()
        
        self.low_dim_coef = IULDPProtocol(feature_index = self.selected_indexes,
                      epsilon = self.epsilon,
                      **self.regressor_params
                    ).apply(X_list[self.n1:], y_list[self.n1:])
        
        self.coef_ = np.zeros(self.d)
        self.coef_[self.selected_indexes] = self.low_dim_coef
        
        return self
    
    def predict(self, X):
        '''
        Predict the label of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted labels.
        '''
        
        X_selected = X[:, self.selected_indexes]
        
        return np.dot(X_selected, self.low_dim_coef).ravel()