import numpy as np


def compute_omega(W : np.ndarray) -> float:
    """ 
    Computes the optimal momentum parameter omega for the gossip matrix W.
    """
    eigenvalues = np.linalg.eigvalsh(W)
    lambda_2 = sorted(eigenvalues)[-2]
    lambda_n = sorted(eigenvalues)[0]
    gamma = 1 - np.maximum(lambda_2, -lambda_n)
    omega = (1 - np.sqrt(gamma*(2-gamma))) / (1 + np.sqrt(gamma*(2-gamma)))
    return omega


def AcceleratedGossip(Y : np.ndarray, W : np.ndarray, L : int, omega : float = 0.) -> np.ndarray:
    """ 
    Performs Accelerated Gossip to average Y_i's using the gossip matrix W for L iterations with parameter omega.
    For omega = 0, this reduces to standard gossip.
    """
    X = Y.copy()
    X_prev = Y.copy()

    for l in range(L):
        X_new = (1 + omega) * np.einsum('ij,jkl->ikl', W, X) - omega * X_prev
        X_prev = X.copy()
        X = X_new.copy()

    return X