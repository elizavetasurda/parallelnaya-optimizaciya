"""
Acquisition functions for Bayesian optimization.
"""

import numpy as np
from scipy.stats import norm

def expected_improvement(mu, sigma, best_y, xi=0.01):
    """
    Expected Improvement acquisition function.
    """
    with np.errstate(divide='warn'):
        imp = mu - best_y - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def upper_confidence_bound(mu, sigma, beta=2.0):
    """
    Upper Confidence Bound acquisition function.
    """
    return mu + beta * sigma