import numpy as np
from scipy.linalg import subspace_angles

def sin_thetak(X : np.ndarray, U : np.ndarray, k : int) -> float:
    """ 
    Compute the sine of the k-th principal angle between the subspaces spanned by the columns of X and U. X and U are column-orthonormal matrices.
    """
    # To handle cases where X or U have more than k columns
    p = min(X.shape[1], U.shape[1])
    return np.sin(subspace_angles(X, U)[p-k])

def largest_spnorm(A : np.ndarray) -> float:
    """ 
    Compute the largest spectral norm among the matrices A[i].
    """
    return max(np.linalg.norm(A[i], ord=2) for i in range(A.shape[0]))

def reconstruction_error(X : np.ndarray, A : np.ndarray) -> float:
    """ 
    Compute the reconstruction error ||A - X X^T A||_F.
    """
    return np.linalg.norm(A - X @ (A @ X).T, ord='F')