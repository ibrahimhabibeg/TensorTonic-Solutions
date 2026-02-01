import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Use numpy
    values = np.array(values)
    transitions = np.array(transitions)
    rewards = np.array(rewards)

    # Calc Reward
    return (rewards + gamma * (transitions * values).sum(axis=-1)).max(axis=-1).tolist()