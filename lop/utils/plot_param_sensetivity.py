import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib as mpl


def generate_parameter_sensitivity_plot(
        final_performances=None,
        param_axis_1=[],
        colors=None,
        yticks=[],
        xticks=[],
        labels=None,
        xlabel='',
        ylabel='',
):
    """
    This function plots the parameter sensitivity plot for various hyper parameter settings.
    It plots the mean and std-error for each configuration
    """
    if colors is None:
        colors = [(0, 1, 0, 1), (0, 0, 1, 1), (0.5, 0.5, 0, 1), (1, 0, 0, 1)]

    fig, ax = plt.subplots()
    mpl.rcParams.update({'font.size': 18})
    for idx_1 in range(len(final_performances)):
        means = np.mean(final_performances[idx_1], axis=1)
        num_runs = final_performances[idx_1].shape[1]
        stds = np.std(final_performances[idx_1], axis=1)/sqrt(num_runs)
        
        x = np.array([i for i in param_axis_1[:len(final_performances[idx_1])]])
        color = colors[idx_1]
        label = ''
        if labels is not None:
            label = labels[idx_1]
        print(str(label))
        plt.plot(x, means, '-', color=color, label=label)
        plt.fill_between(x, means - stds, means + stds, alpha=0.2, color=color)

    plt.xscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=12)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=12)
    ax.set_ylim(yticks[0], yticks[-1])

    ax.yaxis.grid()

    plt.legend(fontsize=10)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.tight_layout()
    plt.savefig('sens_plot.png', dpi=500)
    plt.close()
