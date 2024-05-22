from .marginal_distributions import (LaplaceDistribution, 
                          BetaDistribution,
                          DeltaDistribution,
                          MultivariateNormalDistribution,
                          MarginalIndependentMultivariateNormalDistribution,
                          ExponentialDecayMultivariateNormalDistribution,
                          UniformDistribution,
                          MarginalDistribution,
                          ExponentialDistribution,
                          MixedDistribution,
                          UniformCircleDistribution,
                          CauchyDistribution,
                          CosineDistribution,
                          TDistribution
                          )

from .regression_function import RegressionFunction

from .noise_distributions import GaussianNoise

from .joint_distribution import JointDistribution

import numpy as np
import math





class TestDistribution(object):
    def __init__(self,index, dim = "auto"):
        self.dim = dim
        self.index = index
        
    def testDistribution_1(self):
        if self.dim == "auto":
            self.dim = 1
        
        marginal_obj = MarginalIndependentMultivariateNormalDistribution(self.dim)
        beta = np.zeros(self.dim)
        beta[:8] = 1 / 5
        regression_obj = RegressionFunction(beta)
        noise_obj = GaussianNoise(1)  
   
        return JointDistribution(marginal_obj, regression_obj, noise_obj)
    
    
    def testDistribution_2(self):
        if self.dim == "auto":
            self.dim = 1
        
        
        related_dim = min(self.dim, 50)
        marginal_obj_1 = ExponentialDecayMultivariateNormalDistribution(related_dim, 2/3)
        marginal_obj_2 = MarginalIndependentMultivariateNormalDistribution(self.dim - related_dim)
        marginal_obj = MarginalDistribution([marginal_obj_1, marginal_obj_2])
        beta = np.zeros(self.dim)
        beta[:8] = 1 / 5
        regression_obj = RegressionFunction(beta)
        noise_obj = GaussianNoise(1)  
   
        return JointDistribution(marginal_obj, regression_obj, noise_obj)
    
    def testDistribution_3(self):
        if self.dim == "auto":
            self.dim = 1
        
        
        related_dim = min(self.dim, 15)
        marginal_obj_1 = ExponentialDecayMultivariateNormalDistribution(related_dim, 1/2)
        marginal_obj_2 = MarginalIndependentMultivariateNormalDistribution(self.dim - related_dim)
        marginal_obj = MarginalDistribution([marginal_obj_1, marginal_obj_2])
        active_dims = np.random.choice(16, size = 8, replace=False)
        beta = np.zeros(self.dim)
        beta[active_dims] = 1 / 5
        regression_obj = RegressionFunction(beta)
        noise_obj = GaussianNoise(1)  
   
        return JointDistribution(marginal_obj, regression_obj, noise_obj)
 
 
        

    def returnDistribution(self):
        switch = {'1': self.testDistribution_1,   
                  '2': self.testDistribution_2,     
                  '3': self.testDistribution_3,             
          }

        choice = str(self.index)  
        result=switch.get(choice)()
        return result
    
