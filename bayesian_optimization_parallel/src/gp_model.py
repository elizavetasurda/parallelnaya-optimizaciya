"""
Gaussian Process model for Bayesian optimization.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class GPModel:
    def __init__(self, kernel=None):
        if kernel is None:
            kernel = Matern(nu=2.5)
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, return_std=True):
        return self.model.predict(X, return_std=return_std)