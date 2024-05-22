'''
Used for first phase variable selection.
Each object is able to select variable based on local samples and reture a index of selected variable.
'''

import numpy as np
import math
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from scipy.stats import norm


class LassoSelector(object):
    def __init__(self, 
                 num_features = 1,
                 screening_num_features = None,
                 alpha = None, 
                 max_iter = None, 
                 selection = None,
                 tol = None):
        '''
        Input:
        ---------
        alpha, the regularization parameter
        max_iter, the maximum number of iterations
        selection, the selection criterion
        tol, the tolerance for lasso
        '''
        self.alpha = alpha
        self.num_features = num_features

        # filter the none value
        params = {
                    "max_iter": max_iter,
                    "selection": selection,
                    "tol": tol}
        self.params = {k: v for k, v in params.items() if v is not None}


    
    def fit(self, X, y):
        '''
        Input:
        ---------
        X, the local samples
        y, the local labels

        Output:
        ---------
        self, the fitted object
        '''
        self.m, self.d = X.shape
        if self.alpha is None:
            clf = LassoCV(**self.params)
        else:
            clf = Lasso(alpha = self.alpha, **self.params)
        clf.fit(X, y)
        self.alpha = clf.alpha_
        
        count = 0
        while True:
            count += 1
            if count > 15:
                break
            
            
            if len(np.nonzero(clf.coef_)[0]) < self.num_features and len(np.nonzero(clf.coef_)[0]) < self.d:
                self.alpha = self.alpha / 2
                clf = Lasso(alpha = self.alpha, **self.params)
                clf.fit(X, y)

            elif len(np.nonzero(clf.coef_)[0]) > 2 * self.num_features:
                self.alpha = self.alpha * 1.5
                clf = Lasso(alpha = self.alpha, **self.params)
                clf.fit(X, y)
            else:
                break
        
        
        
        self.indexes = np.nonzero(clf.coef_)[0]
        add_idx = 0
        while len(self.indexes) < self.num_features and len(self.indexes) < self.d and add_idx < self.d:
            if add_idx not in self.indexes:
                self.indexes = np.append(self.indexes, add_idx)
            add_idx += 1
            
        return self
    
    def get_idx(self, X = None, y = None):
        '''
        Input:
        ---------
        X, the local samples
        y, the local labels

        Output:
        ---------
        idx, the selected index
        '''
        if not hasattr(self, 'indexes'):
            assert X is not None and y is not None, "Please fit the model or input X and y"
            self.fit(X, y)
        return np.random.choice(self.indexes, 1)[0]
    

class ScreeningSelector(object):
    def __init__(self, 
                 num_features = 2,
                 screening_num_features = None,
                 alpha = None, 
                 max_iter = None, 
                 selection = None,
                 tol = None
                 ):
        '''
        Input:
        ---------
        num_features, the number of features to be selected
        '''
        
        self.num_features = num_features
        
        
    
    def fit(self, X, y):
        '''
        Input:
        ---------
        X, the local samples
        y, the local labels

        Output:
        ---------
        self, the fitted object
        '''
        # compute the marginal correlation
        self.m, self.d = X.shape
        self.corr = (y @ X ).ravel()

        # the i th quantile value of corr
        quantile = np.quantile(self.corr, max(1 - self.num_features / self.d, 0))
        self.indexes = np.nonzero(self.corr >= quantile)[0]
        return self
    
    def get_idx(self, X = None, y = None):
        '''
        Input:
        ---------
        X, the local samples
        y, the local labels

        Output:
        ---------
        idx, the selected index
        '''

        if not hasattr(self, 'indexes'):
            assert X is not None and y is not None, "Please fit the model or input X and y"
            self.fit(X, y)
        return np.random.choice(self.indexes, 1)[0]
    


class PostLassoSelector(object):
    def __init__(self, 
                 num_features = 1,
                 screening_num_features = 4,
                 alpha = None, 
                 max_iter = None, 
                 selection = None,
                 tol = None):
        '''
        Input:
        ---------
        alpha, the regularization parameter
        max_iter, the maximum number of iterations
        selection, the selection criterion
        tol, the tolerance for lasso
        '''
        self.lasso_selector = LassoSelector(num_features = num_features,
                                            alpha = alpha,
                                            max_iter = max_iter,
                                            selection = selection,
                                            tol = tol)
        
        self.screening_selector = ScreeningSelector(num_features = screening_num_features)

    def fit(self, X, y):
        '''
        Input:
        ---------
        X, the local samples
        y, the local labels

        Output:
        ---------
        self, the fitted object
        '''
        # screening selection
        screening_indexes = self.screening_selector.fit(X, y).indexes
        X = X[:, screening_indexes]
        # post selection
        self.indexes = screening_indexes[self.lasso_selector.fit(X, y).indexes]

        
        return self
    
    def get_idx(self, X = None, y = None):
        '''
        Input:
        ---------
        X, the local samples
        y, the local labels

        Output:
        ---------
        idx, the selected index
        '''
        if not hasattr(self, 'indexes'):
            assert X is not None and y is not None, "Please fit the model or input X and y"
            self.fit(X, y)
        return np.random.choice(self.indexes, 1)[0]

