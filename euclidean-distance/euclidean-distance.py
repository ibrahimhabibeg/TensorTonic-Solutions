import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    return ((np.array(x) - np.array(y)) ** 2).sum() ** 0.5