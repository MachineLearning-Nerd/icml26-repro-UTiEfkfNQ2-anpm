from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

import matplotlib.pyplot as plt
from anpm.plotting.plot import plot_ax


def main():
    output_dir = "plots"
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(1, 2, figsize=(6.7, 3), sharex=True, sharey=True)
    xticks = [0, 25, 50, 75, 100]

    # (Left)  Small noise
    csv_path_beta_smallnoise = Path("results/anpm_amazon_sigma1e-3_k30_every10.csv")
    colors_beta = ["#9FC1EF", "#FFAF3F"]
    linestyle_beta = ["-"] * 2
    markers_beta = ["o", "s"]
    plot_ax(ax[0], csv_path_beta_smallnoise, colors_beta, linestyle_beta, 
            markers=markers_beta, markers_space=1, 
            legend = False, x_ticks = xticks, 
            y_label=r'$\Vert(\mathbf{I}_d - \mathbf{X}_t\mathbf{X}_t^\top)\mathbf{A}\Vert_2 - \lambda_{k+1}(\mathbf{A})$', x_label = r'$t$')

    # (Right) Large noise
    csv_path_beta_largenoise = Path("results/anpm_amazon_sigma2e-3_k30_every10.csv")
    plot_ax(ax[1], csv_path_beta_largenoise, colors_beta, linestyle_beta, markers=markers_beta, markers_space=1, legend = True, legend_loc = "lower right", legend_ncol = 2, legend_title = r"$\beta$", legend_box = True, x_ticks = xticks, x_label= r'$t$')

    plt.tight_layout()
    output_path = output_dir / "figure4.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()