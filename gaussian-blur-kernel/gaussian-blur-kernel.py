import numpy as np

def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    x = np.repeat(np.expand_dims(np.arange(size) - size//2, axis=0), size, axis=0) # (size, size)
    y = np.repeat(np.expand_dims(np.arange(size) - size//2, axis=1), size, axis=1) # (size, size)
    kernel = np.exp(-(x**2 + y**2) / (2*sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.tolist()

