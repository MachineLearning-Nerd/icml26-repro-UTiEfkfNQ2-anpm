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
    parser.add_argument('--gap', type=float, default=1e-2)
    parser.add_argument('--noise_type', type=str, default='adversarial', choices=['adversarial', 'stochastic'])
    parser.add_argument('--exp_name', type=str, default='')
    args = parser.parse_args()
    return args.d, args.k, args.T, args.gap, args.noise_type, args.exp_name

def main():
    np.random.seed(0)

    d, k, T, gap, noise_type, exp_name = parser()

    U = generate_eigenvectors(d)
    lambda_1 = 1.
    lambda_k = 1.0
    lambda_kp1 = 1.0 - gap
    lambda_rest = .5
    lambdas = np.array([lambda_1]*(k-1) 
                    + [lambda_k] 
                    + [lambda_kp1] 
                    + [lambda_rest]*(d - k -1))
    A = generate_matrix(lambdas, U)
    X0 = generate_X0(d, k)
    beta = lambda_kp1**2/4


    errors_dict = {}

    xis = [10**(-i) for i in np.linspace(2, 4, 6)]
    labels = [r'$10^{{-{}}}$'.format(i) for i in np.linspace(2, 4, 6)]

    for i, xi in enumerate(xis):
        Xi = generate_noise_type(noise_type, d, k, T, xi, U)
        X_list = anpm(A, beta, T, X0, Xi)
        errors = [sin_thetak(U[:, :k], X[:,:k], k) for X in X_list]
        errors_dict[labels[i]] = errors


    n_rows = len(next(iter(errors_dict.values())))

    result_path = f"results/anpm_synthetic_noise_{exp_name}.csv"
    with open(result_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t"] + list(errors_dict.keys())
        writer.writerow(header)
        for t in range(n_rows):
            row = [t] + [errors_dict[name][t] for name in errors_dict.keys()]
            writer.writerow(row)

if __name__ == "__main__":
    main()