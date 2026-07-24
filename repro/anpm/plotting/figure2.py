from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from anpm.plotting.plot import plot_ax


def main():
    output_dir = "plots"
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(2, 2, figsize=(6.7, 4.0), sharex=True, sharey=False)
    xticks = [0, 50, 100, 150, 200]

    colors = ["#1DBB30","#435CFF","#FFD900","#FF0D00"] * 2
    linestyle = ["-"] * 4 + ["-."] * 4
    markers = ["s", "+", "*", "x"] * 2
    alpha = .8

    # (Upper left)  Fed-Heart-Disease
    csv_path_beta_large = Path("results/depca_heart_.csv")
    plot_ax(ax[0, 0], csv_path_beta_large, colors, linestyle, markers,
        alpha = alpha, legend = False, x_ticks = xticks, y_label=r'$\frac{1}{n}\sum\sin\theta_k(\mathbf{U}_k, \mathbf{X}_{i,t})$',
            add_text = True, text = "Fed-Heart-Disease", text_loc = (.57, .9), text_fontsize = 8)

    # (Upper right) Ego-Facebook
    csv_path_ego_facebook = Path("results/depca_ego_facebook_.csv")
    plot_ax(ax[0, 1], csv_path_ego_facebook, colors, linestyle, markers, alpha = alpha, legend = False, x_ticks = xticks,
            add_text = True, text = "Ego-Facebook", text_loc = (.649, .9), text_fontsize = 8)

    # (Bottom left) Digits homogeneous
    csv_path_gap = Path("results/depca_digits_hom.csv")
    plot_ax(ax[1, 0], csv_path_gap, colors, linestyle, markers, alpha = alpha, x_label=r'$t$', y_label=r'$\frac{1}{n}\sum\sin\theta_k(\mathbf{U}_k, \mathbf{X}_{i,t})$', legend = False, x_ticks = xticks,
            add_text = True, text = r"Digits homogeneous, $M \approx 230$", text_loc = (.325, .9), text_fontsize = 8)

    # (Bottom right) Digits heterogeneous
    csv_path_noise = Path("results/depca_digits_het.csv")
    plot_ax(ax[1, 1], csv_path_noise, colors, linestyle, markers, alpha = alpha, x_ticks=xticks, x_label = r'$t$', legend = False,
            add_text = True, text = r"Digits heterogeneous, $M \approx 725$", text_loc = (.31, .9), text_fontsize = 8)
    
    # Legend : L
    L_labels = [r'$L=20$', r'$L=40$'] 
    L_handles = [Line2D([0], [0], color="black", linestyle=ls, linewidth=1.2, label=L_labels[i]) for i, ls in enumerate(["-", "-."])]
    ax[1, 0].legend(handles=L_handles, loc="lower left", fontsize=8, frameon=True, ncol=1, facecolor="white", edgecolor="0.5", framealpha=0.85, fancybox=False, borderpad=0.4, handlelength=1.6, handletextpad=0.5, columnspacing=0.8, labelspacing=0.35)      

    # Legend : algorithms
    alg_labels = ['DePM', 'DeEPCA', r'ADePM $\beta=\beta^*$' , r'ADePM $\beta=\beta_t$']
    alg_handles = [Line2D([0], [0], color=colors[i], marker=markers[i], linestyle="-", linewidth=1.2, label=alg_labels[i]) for i in range(4)]
    ax[1, 1].legend(handles=alg_handles,loc="lower left",fontsize=8,frameon=True,ncol=2, facecolor="white", edgecolor="0.5", framealpha=0.85, fancybox=False, borderpad=0.4, handlelength=1.6, handletextpad=0.5, columnspacing=0.8, labelspacing=0.35)

    
    plt.tight_layout()
    output_path = output_dir / "figure2.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()