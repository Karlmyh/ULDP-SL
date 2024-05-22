import numpy as np


class RegressionFunction(object): 
    def __init__(self,beta):
        if len(beta.shape) != 1:
            beta = beta.ravel()
        self.beta = beta
        self.dim = beta.shape[0]
        
    def functional(self, x):
        return np.dot(self.beta, x)
    
    def apply(self, X):
        return np.array([self.functional(data) for data in X])
    

        