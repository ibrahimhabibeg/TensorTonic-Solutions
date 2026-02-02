import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    N, D = X.shape
    assert labels.shape[0] == N, "The number of labels mismatches the number of rows"
    K = len(np.unique(labels))
    assert labels.max() < K and labels.min() >= 0, f"Expected values in labels to be in range [0, {K})"
    

    # Calculate mask (N x N x K) such that 
    # mask[i, j, k] = 1 if the distance from i to j is considered when calculating
    #                   the average distance from i to cluster k
    #                 0 otherwise
    # mask[i, j, k] = 0 if i == j
    #                 1 if label[j] == k
    #                 0 otherwise

    unique_labels = np.unique(labels) # (K) 
    mask = np.expand_dims(labels, axis=-1) == np.expand_dims(unique_labels, axis=0) # (N, K)
    mask = np.repeat(np.expand_dims(mask, axis=0), N, axis=0) # (N, N, K)
    mask[np.arange(N), np.arange(N), :] = 0 # (N, N, K)
    
    # Calculate Euclidean distance matrix (N x N) showing the distance between each pair
    diff = np.expand_dims(X, axis=1) - np.expand_dims(X, axis=0) # (N, N, D)
    distances = (diff ** 2).sum(axis=-1) ** 0.5 # (N, N)

    # Calculate the average distance distance to cluster (N x K)
    sum_distance_to_cluster = (np.expand_dims(distances, axis=-1) * mask).sum(axis=1) # (N x K)
    count_used_in_cluster_calculation = mask.sum(axis=1) # (N x K)
    avg_distance_to_cluster = sum_distance_to_cluster / count_used_in_cluster_calculation # (N x K)

    # Calculate cohesion
    cohesion = avg_distance_to_cluster[np.arange(N), labels] # (N)

    # Calculate seperation
    masked_avg_distance_to_cluster = avg_distance_to_cluster.copy()
    masked_avg_distance_to_cluster[np.arange(N), labels] = np.inf
    seperation = masked_avg_distance_to_cluster.min(axis=-1) # (N)

    # Calculate Silhouette
    silhouette = (seperation - cohesion) / np.maximum(seperation, cohesion) # (N)

    return silhouette.mean()
