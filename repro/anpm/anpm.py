import numpy as np

def anpm(A : np.ndarray, beta : float, T : int, X0 : np.ndarray, Xi : np.ndarray) -> list[np.ndarray]:
    """ 
    Performs Accelerated Noisy Power Method to compute the top-k eigenspace of a PSD matrix A.
    """
    X_prev = np.copy(X0)
    Y = 1/2 * A @ X0 + Xi[0]
    X, R = np.linalg.qr(Y, mode='reduced')

    X_list = [X0, X]

    for t in range(1, T):
        xi = Xi[t]
        Y = A @ X - beta * X_prev @ np.linalg.inv(R) + xi
        X_new, R = np.linalg.qr(Y, mode='reduced')
        X, X_prev = X_new, X
        X_list.append(X)

    return np.array(X_list)


def anpm_tune(A : np.ndarray, T : int, X0 : np.ndarray, Xi : np.ndarray) -> list[np.ndarray]:
    """ 
    Performs Accelerated Noisy Power Method to compute the top-k eigenspace of a PSD matrix A. Momentum parameter tuned adaptively.
    """
    X_prev = np.copy(X0)
    Y = 1/2 * A @ X0 + Xi[0]
    X, R = np.linalg.qr(Y, mode='reduced')
    
    X_list = [X0, X]

    for t in range(1, T):
        xi = Xi[t]
        beta = np.sort(np.diag(X.T @ (A @ X + xi)))[0]**2 / 4
        Y = A @ X - beta * X_prev @ np.linalg.inv(R) + xi
        X_new, R = np.linalg.qr(Y, mode='reduced')
        X, X_prev = X_new, X
        X_list.append(X)

    return np.array(X_list)