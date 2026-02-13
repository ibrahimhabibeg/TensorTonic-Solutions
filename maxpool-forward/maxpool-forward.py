import numpy as np

def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    X = np.array(X)
    H, W = X.shape
    W_out = (W-pool_size)//stride + 1
    H_out = (H-pool_size)//stride + 1
    ans = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            ans[i][j] = np.max(X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size])
    return ans.tolist()