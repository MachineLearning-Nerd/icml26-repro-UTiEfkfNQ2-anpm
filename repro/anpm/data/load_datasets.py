import numpy as np
import scipy.sparse as sp

from anpm.data.graphs import RingGraph

def load_fed_heart_disease():
    from flamby.datasets.fed_heart_disease import FedHeartDisease

    n = 4
    d = 13
    total_samples = 486

    A = np.zeros((n, d, d))
    for i in range(n):
        center = FedHeartDisease(center=i, train=True)
        X = (np.array(center.features) - np.array(center.mean_of_features)[0, 0]) / (np.array(center.std_of_features)[0, 0] + 1e-8)
        A[i] = (n / total_samples) * (X.T @ X)

    A_mean = np.mean(A, axis=0)

    W = RingGraph(n)
    
    return A, W, A_mean


def load_ego_facebook(n : int):
    import networkx as nx

    path = "datasets/facebook_combined.txt"
    G = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int)
    G = G.subgraph(range(n))
    
    # Constructing the local matrices A_i so that sum_i A_i = 2*I_n - L_norm
    A = np.zeros((n, n, n))
    for i in range(n):
        neighbors = list(G.neighbors(i))
        deg_i = len(neighbors)
        for j in neighbors:
            deg_j = G.degree[j]
            A[i, i, j] = 1/2/np.sqrt(deg_i * deg_j)
            A[i, j, i] = A[i, i, j]
        A[i, i, i] = 1
    A = n * A

    A_mean = np.mean(A, axis=0)

    # We add ~n log(n) edges to the graph to increase connectivity for the gossip matrix
    G_prime = G.copy()
    extra_edges = n * int(np.log(n))
    possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not G_prime.has_edge(i, j)]
    np.random.shuffle(possible_edges)
    for edge in possible_edges[:extra_edges]:
        G_prime.add_edge(edge[0], edge[1])

    # Constructing the gossip matrix W using metropolis weights
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = list(G_prime.neighbors(i))
        deg_i = len(neighbors)
        W[i,i] = 1
        for j in neighbors:
            deg_j = G_prime.degree[j]
            W[i,j] = 1 / (1 + max(deg_i, deg_j))
            W[i,i] -= W[i,j]

    return A, W, A_mean


def load_digits_homogeneous():
    from sklearn.datasets import load_digits

    n = 10
    d = 64

    digits = load_digits()
    indices = np.arange(len(digits.data))
    m = len(digits.data)
    np.random.shuffle(indices)

    A = np.zeros((n, d, d))
    mean = np.mean(digits.data, axis=0)
    for i in range(n):
        if i < n-1:
            X_i = digits.data[indices[i*180:(i+1)*180], :]
        else:
            X_i = digits.data[indices[i*180:], :]
        X_i = (X_i - mean)
        A[i] = (n / m) * (X_i.T @ X_i)
    
    A_mean = np.mean(A, axis=0)

    W = RingGraph(n)

    return A, W, A_mean


def load_digits_heterogeneous():
    from sklearn.datasets import load_digits

    n = 10
    d = 64

    digits = load_digits()
    y = digits.target
    m = len(digits.data)

    A = np.zeros((n, d, d))
    mean = np.mean(digits.data, axis=0)
    for i in range(n):
        X_i = digits.data[y == i]
        X_i = (X_i - mean)
        A[i] = (n / m) * (X_i.T @ X_i)
    
    A_mean = np.mean(A, axis=0)

    W = RingGraph(n)

    return A, W, A_mean

def load_amazon(path: str = "datasets/amazon0302.txt"):
    n_from_header = None
    with open(path) as f:
        for line in f:
            if line.startswith("# Nodes:"):
                n_from_header = int(line.split()[2])
                break

    edges = np.loadtxt(path, dtype=np.int64, comments="#")
    row = edges[:, 0]
    col = edges[:, 1]
    n = max(int(max(row.max(), col.max())) + 1, n_from_header or 0)

    rows = np.concatenate((row, col))
    cols = np.concatenate((col, row))
    data = np.ones(rows.shape[0], dtype=np.float64)
    S = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    S.sum_duplicates()
    S.data.fill(1.0)

    degrees = np.asarray(S.sum(axis=1)).ravel()
    inv_sqrt_degrees = np.zeros_like(degrees, dtype=np.float64)
    nonzero = degrees > 0
    inv_sqrt_degrees[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

    A = S.tocoo(copy=False)
    A.data *= inv_sqrt_degrees[A.row] * inv_sqrt_degrees[A.col]
    A = A.tocsr()
    A.setdiag(A.diagonal() + 1.0)
    return A
