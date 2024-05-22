
import numpy as np


class JointDistribution(object): 
    def __init__(self, marginal_obj, regression_obj, noise_obj):
        self.marginal_obj = marginal_obj
        self.regression_obj = regression_obj
        self.noise_obj = noise_obj
        self.beta = self.regression_obj.beta
        self.nonzero_index = np.where(self.beta != 0)[0]

    def generate(self, n):
        X = self.marginal_obj.generate(n)
        Y_true = self.regression_obj.apply(X)
        return X, Y_true+ self.noise_obj.generate(n)
    
    def generat_true(self, n):
        X = self.marginal_obj.generate(n)
        Y_true = self.regression_obj.apply(X)
        return X, Y_true

    def evaluate(self, X):
        return self.regression_obj.apply(X)
        