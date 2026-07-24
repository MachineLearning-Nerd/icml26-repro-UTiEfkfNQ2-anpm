import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],

    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
})


def plot_ax(ax : plt.Axes, csv_path : Path, colors, linestyles, 
            markers = None, markers_space = 50, alpha = 1.,
            x_ticks = None, x_label = None, y_label = None,
            legend = True, legend_ncol = 2, legend_loc = "lower right", legend_title = None, legend_box = False,
            add_text = False, text = "", text_loc = (0.05, 0.95), text_fontsize = 11,
            inset = False, inset_xlim = (0, 100), inset_ylim_factor = 0.2):
    
    df = pd.read_csv(csv_path)

    headers = df.columns.tolist()
    labels = headers[1:]
    x = df[headers[0]].values

    if markers is not None:
        for column, color, ls, marker in zip(labels, colors, linestyles, markers):
            ax.plot(x, df[column], label=column, color=color, linewidth=1.2, linestyle=ls, marker=marker, markersize=5, markevery=markers_space, alpha=alpha)
    else:
        for column, color, ls in zip(labels, colors, linestyles):
            ax.plot(x, df[column],label=column, color=color, linewidth=1.2, linestyle=ls, alpha=alpha)
        
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yscale("log")

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if add_text:
        ax.text(text_loc[0], text_loc[1], text, transform=ax.transAxes, fontsize=text_fontsize, verticalalignment="top", bbox=dict(facecolor="white", edgecolor="0.5", alpha=0.85, linewidth=0.8, boxstyle="square,pad=0.35"))

    if legend:
        leg = ax.legend(loc=legend_loc, frameon=legend_box, ncol=legend_ncol, title=legend_title, facecolor="white", edgecolor="0.5", framealpha=0.85, fancybox=False, borderpad=0.4, handlelength=1.6, handletextpad=0.5, columnspacing=0.8, labelspacing=0.35)

    if legend_box:
        leg.get_frame().set_linewidth(0.8)

    if inset:
        axins = inset_axes(ax, width="35%", height="40%", loc="upper right", bbox_to_anchor=(- 0.05, -0.1, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        for column, color, ls in zip(labels, colors, linestyles):
            axins.plot(x,df[column],color=color,linewidth=.65,linestyle=ls)
        axins.set_xlim(*inset_xlim)
        mask = x <= inset_xlim[1]
        ymin = df.loc[mask, labels].min().min()
        ymax = df.loc[mask, labels].max().max()
        axins.set_ylim((1 - inset_ylim_factor) * ymin, (1 + inset_ylim_factor) * ymax)
        axins.set_yscale("log")
        axins.grid(True, alpha=0.25)
        axins.tick_params(axis="both", labelsize=8)
        mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5", linewidth=0.8, alpha=0.7)
