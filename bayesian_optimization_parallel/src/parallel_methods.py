"""
Parallel methods for Bayesian optimization.
"""

import numpy as np
from .acquisition import expected_improvement

def batch_selection(gp_model, X_candidates, batch_size, best_y):
    """
    Select a batch of points using greedy selection.
    """
    selected = []
    for _ in range(batch_size):
        mu, sigma = gp_model.predict(X_candidates, return_std=True)
        ei = expected_improvement(mu, sigma, best_y)
        idx = np.argmax(ei)
        selected.append(X_candidates[idx])
        X_candidates = np.delete(X_candidates, idx, axis=0)
    return np.array(selected)