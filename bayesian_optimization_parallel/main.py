#!/usr/bin/env python3
"""
Main entry point for Bayesian Optimization Parallel project.
"""

import numpy as np
from src.gp_model import GPModel
from src.acquisition import expected_improvement
from src.parallel_methods import batch_selection
from src.test_functions import rosenbrock
from src.utils import plot_convergence, save_results

def optimize_function(func, bounds, n_init=5, n_iter=10, batch_size=2):
    """
    Run Bayesian optimization.
    """
    dim = len(bounds)
    X = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_init, dim))
    y = np.array([func(x) for x in X])

    gp = GPModel()
    gp.fit(X, y)

    for i in range(n_iter):
        # Generate candidates
        X_candidates = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (100, dim))
        batch = batch_selection(gp, X_candidates, batch_size, np.min(y))
        y_batch = np.array([func(x) for x in batch])
        X = np.vstack([X, batch])
        y = np.concatenate([y, y_batch])
        gp.fit(X, y)
        print(f"Iter {i+1}: Best y = {np.min(y)}")

    return X, y

if __name__ == "__main__":
    bounds = [(-2, 2), (-1, 3)]  # for Rosenbrock
    X, y = optimize_function(rosenbrock, bounds)
    plot_convergence(X, y)
    save_results(X, y, 'results/rosenbrock_results.npz')