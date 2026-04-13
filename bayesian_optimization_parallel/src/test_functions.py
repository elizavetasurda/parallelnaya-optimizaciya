"""
Test functions for optimization.
"""

import numpy as np

def rosenbrock(x):
    """
    Rosenbrock function.
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def branin(x):
    """
    Branin function.
    """
    x1, x2 = x
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s