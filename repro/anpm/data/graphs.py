import numpy as np

def RingGraph(n : int) -> np.ndarray:
    """ 
    Generate the gossip matrix W for a ring graph with n nodes using Metropolis weights.
    """
    W = np.zeros((n, n))
    for i in range(n):
        W[i, i] = 1/2
        W[i, (i-1)%n] = 1/4
        W[i, (i+1)%n] = 1/4
    return W