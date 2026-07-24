from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

import matplotlib.pyplot as plt
from anpm.plotting.plot import plot_ax


def main():
    output_dir = "plots"
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(2, 2, figsize=(6.7, 4.0), sharex=True, sharey=True)
    xticks = [0, 250, 500, 750, 1000]

    # (Upper left)  Beta, large gap
    csv_path_beta_large = Path("results/anpm_synthetic_beta_largegap.csv")
    n_beta = 8
    colors_beta = ["#C3DBFF","#9FC1EF","#7BA6DF","#578CCE","#3371BE", "#3D4FA2","#0A075C", "#FFAF3F"]
    linestyle_beta = ["-"] * (n_beta - 1) + ["--"]
    plot_ax(ax[0, 0], csv_path_beta_large, colors_beta, linestyle_beta, legend = False, inset = True, x_ticks = xticks, y_label=r'$\sin\theta_k(\mathbf{U}_k, \mathbf{X}_t)$')

    # (Upper right) Beta, small gap
    csv_path_beta_small = Path("results/anpm_synthetic_beta_smallgap.csv")
    plot_ax(ax[0, 1], csv_path_beta_small, colors_beta, linestyle_beta, legend = True, legend_loc = "lower right", legend_ncol = 2, legend_title = r"$\beta$", legend_box = True, x_ticks = xticks)

    # (Bottom left) Gap
    csv_path_gap = Path("results/anpm_synthetic_gap_.csv")
    n_gap = 6
    colors_gap = ["#FFC0AA","#FA9C8A","#F5786A","#F0534B","#EB2F2B","#E60B0B"]
    linestyle_gap = ["-"] * n_gap
    plot_ax(ax[1, 0], csv_path_gap, colors_gap, linestyle_gap, x_label=r'$t$', y_label=r'$\sin\theta_k(\mathbf{U}_k, \mathbf{X}_t)$',
            legend = True, legend_loc = "upper right", legend_ncol = 1, legend_title = r"$\Delta_k$", legend_box = True, x_ticks = xticks)

    # (Bottom right) Noise
    csv_path_noise = Path("results/anpm_synthetic_noise_.csv")
    n_noise = 6
    colors_noise = ["#9BD78D","#7FBC72","#63A257","#46873B","#2A6D20","#0E5205"]
    linestyle_noise = ["-"] * n_noise
    plot_ax(ax[1, 1], csv_path_noise, colors_noise, linestyle_noise, x_ticks=xticks, x_label = r'$t$',
            legend = True, legend_loc = "upper right", legend_ncol = 1, legend_title = r"$\xi$", legend_box = True)
    
    plt.tight_layout()
    output_path = output_dir / "figure1.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()