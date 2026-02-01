import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.array(X)
    if X.ndim != 2:
        return None # Expect a 2D array

    N, D = X.shape
    if N < 2:
        return None # Can't calaculate correlation from only one sample

    sigma = 1/(N-1) * (X - X.mean(axis=0)).T @ (X - X.mean(axis=0))
    # sigma_{i,j} = (The Correlation of the i^th and j^th features; the variance when i == j)
    return sigma