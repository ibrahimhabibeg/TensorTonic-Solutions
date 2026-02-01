import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    points = np.array(points)
    T = np.array(T)

    # Ensure we are working on a batch
    if points.ndim == 1:
        points = np.expand_dims(points, axis=0)

    N, d = points.shape
    assert d == 3, "Expected the points to be 3D"
    
    homogeneous = np.concatenate((points.T, np.ones((1, N))), axis=0) # (4, N)
    transformed_homogeneous = T @ homogeneous # (4, N)

    return transformed_homogeneous[:3, :].T.squeeze() # (N, 3) or (3)
    