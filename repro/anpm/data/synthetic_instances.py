import numpy as np

def generate_noise(d : int, k : int, T : int, eta : float) -> np.ndarray:
    """ 
    Sample T d x k matrices uniformly on the sphere of radius eta for the spectral norm
    """
    Z = np.random.normal(size=(T, d, k))
    for t in range(T):
        Z[t] /= np.linalg.norm(Z[t], ord=2)
    return eta * Z

def generate_adversarial_noise(d : int, k : int, T : int, eta : float, U : np.ndarray) -> np.ndarray:
    Z = np.random.randn(T, d, k)
    for t in range(T):
        Z[t] = np.abs(Z[t])
        Z[t] /= np.linalg.norm(Z[t], ord=2)
        Z[t] = U @ Z[t]
    return - eta * Z

def generate_gaussian_noise(d : int, k : int, T : int, eta : float) -> np.ndarray:
    Z = np.random.normal(size=(T, d, k))
    return eta * Z


def generate_eigenvectors(d : int) -> np.ndarray:
    """ 
    Uniformly sample a d x d orthonormal matrix
    """
    Q, _ = np.linalg.qr(np.random.normal(size=(d, d)))
    return Q


def generate_X0(d : int, k : int) -> np.ndarray:
    """ 
    Generate initial d x k column-orthonormal matrix
    """
    X0, _ = np.linalg.qr(np.random.normal(size=(d, k)))
    return X0


def generate_matrix(lambdas : np.ndarray, Q : np.ndarray) -> np.ndarray:
    """ 
    Generate a d x d PSD matrix with eigenvalues lambdas and eigenvectors Q
    """
    return Q @ np.diag(lambdas) @ Q.T
