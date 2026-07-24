from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator, eigsh

from anpm.data.synthetic_instances import generate_X0
from anpm.data.load_datasets import load_amazon

import argparse
import csv

    
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--sigma', type=float, default=1e-4)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--error_tol', type=float, default=1e-3)
    parser.add_argument('--error_maxiter', type=int, default=100)
    parser.add_argument('--baseline_tol', type=float, default=1e-6)
    parser.add_argument('--baseline_maxiter', type=int, default=1000)
    parser.add_argument('--error_every', type=int, default=1)
    args = parser.parse_args()
    return (
        args.T,
        args.k,
        args.sigma,
        args.exp_name,
        args.error_tol,
        args.error_maxiter,
        args.baseline_tol,
        args.baseline_maxiter,
        args.error_every,
    )


def gaussian_noise(seed, d, k_total, k, sigma):
    noise = np.random.default_rng(seed).normal(size=(d, k_total))
    return sigma * noise[:, :k]


def momentum_term(X_prev, R):
    return solve_triangular(R.T, X_prev.T, lower=True, check_finite=False).T


def approximation_error(A, X, tol, maxiter, v0=None):
    d = A.shape[0]

    def matvec(v):
        Av = A @ v
        residual = Av - X @ (X.T @ Av)
        return A @ residual

    normal_operator = LinearOperator(
        shape=(d, d),
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = eigsh(
        normal_operator,
        k=1,
        which='LA',
        tol=tol,
        maxiter=maxiter,
        v0=v0,
    )
    return np.sqrt(max(eigenvalues[0], 0.0)), eigenvectors[:, 0]


def best_rank_k_error(A, k, tol, maxiter):
    eigenvalues = eigsh(
        A,
        k=k + 1,
        which='LA',
        return_eigenvectors=False,
        tol=tol,
        maxiter=maxiter,
    )
    return np.sort(eigenvalues)[-k - 1]


def record_error(A, X, best_error, tol, maxiter, v0):
    error, v0 = approximation_error(A, X, tol, maxiter, v0)
    return error - best_error, v0


def run_anpm_errors(
    A,
    beta,
    T,
    X0,
    k,
    noise_seeds,
    sigma,
    best_error,
    error_tol,
    error_maxiter,
    error_every,
):
    d, k_total = X0.shape
    X0 = X0[:, :k]
    X_prev = np.copy(X0)
    v0 = None

    xi = gaussian_noise(noise_seeds[0], d, k_total, k, sigma)
    Y = 0.5 * (A @ X0) + xi
    X, R = np.linalg.qr(Y, mode='reduced')
    error, v0 = record_error(A, X0, best_error, error_tol, error_maxiter, v0)
    errors = [error]
    if error_every == 1:
        error, v0 = record_error(A, X, best_error, error_tol, error_maxiter, v0)
        errors.append(error)

    for t in range(1, T):
        xi = gaussian_noise(noise_seeds[t], d, k_total, k, sigma)
        Y = A @ X - beta * momentum_term(X_prev, R) + xi
        X_new, R = np.linalg.qr(Y, mode='reduced')
        X, X_prev = X_new, X
        iteration = t + 1
        if iteration % error_every == 0:
            error, v0 = record_error(A, X, best_error, error_tol, error_maxiter, v0)
            errors.append(error)

    return errors


def run_anpm_tune_errors(
    A,
    T,
    X0,
    k,
    noise_seeds,
    sigma,
    best_error,
    error_tol,
    error_maxiter,
    error_every,
):
    d, k_total = X0.shape
    X_prev = np.copy(X0)
    v0 = None

    xi = gaussian_noise(noise_seeds[0], d, k_total, k_total, sigma)
    Y = 0.5 * (A @ X0) + xi
    X, R = np.linalg.qr(Y, mode='reduced')
    error, v0 = record_error(A, X0[:, :k], best_error, error_tol, error_maxiter, v0)
    errors = [error]
    if error_every == 1:
        error, v0 = record_error(A, X[:, :k], best_error, error_tol, error_maxiter, v0)
        errors.append(error)

    for t in range(1, T):
        xi = gaussian_noise(noise_seeds[t], d, k_total, k_total, sigma)
        AX = A @ X
        beta = np.min(np.diag(X.T @ (AX + xi))) ** 2 / 4
        Y = AX - beta * momentum_term(X_prev, R) + xi
        X_new, R = np.linalg.qr(Y, mode='reduced')
        X, X_prev = X_new, X
        iteration = t + 1
        if iteration % error_every == 0:
            error, v0 = record_error(A, X[:, :k], best_error, error_tol, error_maxiter, v0)
            errors.append(error)

    return errors

def main():
    np.random.seed(0)

    (
        T,
        k,
        sigma,
        exp_name,
        error_tol,
        error_maxiter,
        baseline_tol,
        baseline_maxiter,
        error_every,
    ) = parser()

    A = load_amazon()
    d = A.shape[0]
    X0 = generate_X0(d, k+1)
    noise_seeds = np.random.default_rng(0).integers(0, 2**63 - 1, size=T)
    best_error = best_rank_k_error(A, k, baseline_tol, baseline_maxiter)
    print(f"Best rank-{k} spectral approximation error: {best_error}")

    betas = [0.0]
    labels = [r'$0$']
    errors_dict = {}

    for beta, label in zip(betas, labels):
        print(f"Running ANPM with beta={beta}...")
        errors_dict[label] = run_anpm_errors(
            A,
            beta,
            T,
            X0,
            k,
            noise_seeds,
            sigma,
            best_error,
            error_tol,
            error_maxiter,
            error_every,
        )

    errors_dict[r'$\beta_t$'] = run_anpm_tune_errors(
        A,
        T,
        X0,
        k,
        noise_seeds,
        sigma,
        best_error,
        error_tol,
        error_maxiter,
        error_every,
    )

    n_rows = len(next(iter(errors_dict.values())))
    t_values = list(range(0, T + 1, error_every))

    result_path = f"results/anpm_amazon_{exp_name}.csv"
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t"] + list(errors_dict.keys())
        writer.writerow(header)
        for t in range(n_rows):
            row = [t_values[t]] + [errors_dict[name][t] for name in errors_dict.keys()]
            writer.writerow(row)
    
if __name__ == "__main__":
    main()
