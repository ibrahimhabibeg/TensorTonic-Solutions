import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N, D = X.shape
    assert N == y.shape[0], "The shape of y mismatches X"

    W = np.zeros((D))
    b = 0.0

    for step in range(steps):
        p = _sigmoid(X@W + b)
        # loss = -(1/N) * (y*np.log(p) + (1-y)*np.log(1-p)).sum()
        grad_b = (1/N) * (p-y).sum()
        grad_W = (1/N) * X.T @ (p-y)
        W = W - lr * grad_W
        b = b - lr * grad_b

    return W, b

