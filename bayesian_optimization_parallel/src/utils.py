"""
Utility functions.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(X, y, title="Convergence"):
    """
    Plot the convergence of the optimization.
    """
    plt.plot(np.minimum.accumulate(y), label='Best so far')
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.title(title)
    plt.legend()
    plt.show()

def save_results(X, y, filename):
    """
    Save results to a file.
    """
    np.savez(filename, X=X, y=y)