from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

import numpy as np
import argparse
import csv

from anpm.depca import *
from anpm.data.load_datasets import *
from anpm.metrics import *
from anpm.gossip import compute_omega

import matplotlib.pyplot as plt

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=200)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--split', type=str, default="hom", choices=["hom", "het"])
    parser.add_argument('--exp_name', type=str, default='hom')
    args = parser.parse_args()
    return args.T, args.k, args.split, args.exp_name

def run_alg(alg, A, T, X0, W, omega, L, beta_star, k):
    if alg == 'DePM':
        return DePM(A, T, X0[:,:k], W, omega, L)
    elif alg == 'DeEPCA':
        return DeEPCA(A, T, X0[:,:k], W, omega, L)
    elif alg == r'ADePM $\beta=\beta^*$':
        return ADePM(A, T+1, beta_star, X0[:,:k], W, omega, L)
    elif alg == r'ADePM $\beta=\beta_t$':
        return ADePM_tune(A, T+1, X0, W, omega, L)


def main():
    
    np.random.seed(0)

    T, k, split, exp_name = parser()
    if split == "hom":
        A, W, A_mean = load_digits_homogeneous()
    else:
        A, W, A_mean = load_digits_heterogeneous()

    eigs, U = np.linalg.eigh(A_mean)
    Uk = U[:, -k:]
    d = A.shape[1]
    n = A.shape[0]

    lambda_k1 = eigs[-k-1]
    beta_star = lambda_k1**2 / 4

    X0 = np.random.randn(d, k+1)
    
    omega = compute_omega(W)
    M = largest_spnorm(A)

    Ls = [20, 40]
    algs = ['DePM', 'DeEPCA', r'ADePM $\beta=\beta^*$' , r'ADePM $\beta=\beta_t$']
    errors_dict = {}

    for L in Ls:
        for alg in algs:
            X_list = run_alg(alg, A, T, X0, W, omega, L, beta_star, k)
            errors = [[sin_thetak(Uk, X[i][:,:k], k) for X in X_list] for i in range(n)]
            errors_mean = np.mean(errors, axis=0)
            errors_dict[alg + ",L=" + str(L)] = errors_mean
    
    n_rows = len(next(iter(errors_dict.values())))

    result_path = f"results/depca_digits_{exp_name}.csv"
    with open(result_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t"] + list(errors_dict.keys())
        writer.writerow(header)
        for t in range(n_rows):
            row = [t] + [errors_dict[name][t] for name in errors_dict.keys()]
            writer.writerow(row)

if __name__ == "__main__":
    main()
