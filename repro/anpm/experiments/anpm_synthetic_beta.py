from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

import numpy as np

from anpm.anpm import anpm, anpm_tune
from anpm.data.synthetic_instances import *
from anpm.metrics import *

import argparse
import csv


def generate_noise_type(noise_type: str, d: int, k: int, T: int, eta: float, U: np.ndarray):
    if noise_type == 'adversarial':
        return generate_adversarial_noise(d, k, T, eta, U)
    else:
        return generate_noise(d, k, T, eta)
    
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=1000)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--noise', type=str, default='adversarial', choices=['adversarial', 'stochastic'])
    parser.add_argument('--gap', type=float, default=0.01)
    parser.add_argument('--xi', type=float, default=1e-4)
    parser.add_argument('--exp_name', type=str, default='largegap')
    args = parser.parse_args()
    return args.d, args.k, args.T, args.noise, args.gap, args.xi, args.exp_name

def main():
    np.random.seed(0)

    d, k, T, noise_type, gap, xi, exp_name = parser()

    U = generate_eigenvectors(d)

    lambda_1 = 1.
    lambda_k = 1.0
    lambda_kp1 = 1.0 - gap
    lambda_rest = 0.5
    lambdas = np.array([lambda_1]*(k-1) 
                       + [lambda_k] 
                       + [lambda_kp1] 
                       + [lambda_rest]*(d - k -1))
    A = generate_matrix(lambdas, U)

    X0 = generate_X0(d, k+1)

    Xi = generate_noise_type(noise_type, d, k+1, T, xi, U)

    errors_dict = {}

    beta_star = (lambda_kp1**2)/4
    beta_crit = lambda_k**2/4
    betas = [0.0, beta_star * 0.5, beta_star * 0.8, beta_star * 0.9, beta_star, 
             0.5 * (beta_star + beta_crit), beta_crit]
    labels = [r'$0$', r'$0.5\beta^\star$', r'$0.8\beta^\star$', r'$0.9\beta^\star$', r'$\beta^\star$', 
              r'$\frac{\beta^\star + \beta_{c}}{2}$', r'$\beta_{c}$']

    for i, beta in enumerate(betas):
        X_list = anpm(A, beta, T, X0, Xi)
        errors = [sin_thetak(U[:, :k], X[:,:k], k) for X in X_list]
        errors_dict[labels[i]] = errors

    X_list = anpm_tune(A, T, X0, Xi)
    errors = [sin_thetak(U[:, :k], X[:,:k], k) for X in X_list]
    errors_dict[r'$\beta_t$'] = errors

    n_rows = len(next(iter(errors_dict.values())))

    result_path = f"results/anpm_synthetic_beta_{exp_name}.csv"
    with open(result_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t"] + list(errors_dict.keys())
        writer.writerow(header)
        for t in range(n_rows):
            row = [t] + [errors_dict[name][t] for name in errors_dict.keys()]
            writer.writerow(row)

if __name__ == "__main__":
    main()