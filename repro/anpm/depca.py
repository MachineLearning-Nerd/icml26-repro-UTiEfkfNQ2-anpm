import numpy as np
from anpm.gossip import *

def ADePM(A : np.ndarray, T : int, beta : float, X0 : np.ndarray, 
          W : np.ndarray, omega : float, L : int):
    """ 
    A is of shape (n, d, d). A[i] is the local data matrix of node i.
    """
    n, _, _ = A.shape

    X = np.array([X0.copy() for _ in range(n)])
    X_prev = X.copy()
    R = np.repeat(np.eye(X0.shape[1])[np.newaxis, :, :], n, axis=0)

    X_list = [X]

    for t in range(1, T):
        if t == 1:
            Y = 1/2 * np.einsum('ijk,ikl->ijl', A, X)
        else:
            Y = np.einsum('ijk,ikl->ijl', A, X) - beta * np.einsum('ijl,ilk->ijk', X_prev, np.linalg.inv(R))
        X_prev = X.copy()
        Y = AcceleratedGossip(Y, W, L, omega)
        X, R = np.linalg.qr(Y, mode='reduced')
        X_list.append(X)
    
    return np.array(X_list)



def ADePM_tune(A : np.ndarray, T : int, X0 : np.ndarray, 
          W : np.ndarray, omega : float, L : int):
    n, _, _ = A.shape

    X = np.array([X0.copy() for _ in range(n)])
    X_prev = X.copy()
    R = np.repeat(np.eye(X0.shape[1])[np.newaxis, :, :], n, axis=0)

    X_list = [X]

    for t in range(1, T):
        if t == 1:
            Y = 1/2 * np.einsum('ijk,ikl->ijl', A, X)
        else:
            # AX = AcceleratedGossip(np.einsum('ijk,ikl->ijl', A, X), W, L, omega)
            AX = Y + np.einsum("i,ijk->ijk",beta, np.einsum('ijl,ilk->ijk', X_prev, np.linalg.inv(R))) if t > 2 else 2 * Y
            XTAX = np.einsum('ijk,ijl->ikl', X, AX)   
            beta = np.sort((np.diagonal(XTAX, axis1=1, axis2=2)), axis=1)[:, 0]**2 * .9 / 4
            Y = np.einsum('ijk,ikl->ijl', A, X) - np.einsum("i,ijk->ijk",beta, np.einsum('ijl,ilk->ijk', X_prev, np.linalg.inv(R)))
        X_prev = X.copy()
        Y = AcceleratedGossip(Y, W, L, omega)            

        # Need R to have positive diagonal coefficients
        X, R = np.linalg.qr(Y, mode='reduced')
        signs = np.sign(np.diagonal(R, axis1=1, axis2=2))
        X = X * signs[:, np.newaxis, :]
        R = R * signs[:, :, np.newaxis]

        X_list.append(X)
    
    return np.array(X_list)



def DePM(A : np.ndarray, T : int, X0 : np.ndarray, 
          W : np.ndarray, omega : float, L : int):
    n, _, _ = A.shape

    X = np.array([X0.copy() for _ in range(n)])

    X_list = [X]

    for t in range(T):
        Y = np.einsum('ijk,ikl->ijl', A, X)
        Y = AcceleratedGossip(Y, W, L, omega)            
        X, _ = np.linalg.qr(Y, mode='reduced')
        X_list.append(X)
    
    return np.array(X_list)



def DeEPCA(A : np.ndarray, T : int, X0 : np.ndarray, 
          W : np.ndarray, omega : float, L : int):
    n, _, _ = A.shape

    X = np.array([X0.copy() for _ in range(n)])
    S_prev = X.copy()
    Y_prev = X.copy()


    X_list = [X]

    for t in range(T):
        Y = np.einsum('ijk,ikl->ijl', A, X)
        S = S_prev + Y - Y_prev
        S = AcceleratedGossip(S, W, L, omega)            
        X, _ = np.linalg.qr(S, mode='reduced')
        for col in range(X.shape[-1]):
            signs = np.sign(np.einsum('ij,ij->i', X[:, :, col], X0[np.newaxis, :, col]))
            X[:, :, col] *= signs[:, np.newaxis]

        X_list.append(X)
        S_prev = S.copy()
        Y_prev = Y.copy()
    
    return np.array(X_list)