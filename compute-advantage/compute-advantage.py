import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    N = len(states)
    states = np.array(states)
    G = np.zeros(N)
    G[N-1] = rewards[N-1]
    for i in range(N-2, -1, -1):
        G[i] = rewards[i] + gamma*G[i+1]
    advantages = G - V
    return advantages

