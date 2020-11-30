from itertools import accumulate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch


def plot_grad_flow(named_parameters_list, out_path):
    """
    Utility function to plot and save gradient magnitude (flow) through a list of interfacing models
    """
    fig0, ax0 = plt.subplots()
    fig0.set_figheight(5)
    fig0.set_figwidth(10)
    ave_grads = []
    layers = []
    counts = [0 for n_p in named_parameters_list]
    for i, named_parameters in enumerate(named_parameters_list):
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n) and (p.grad is not None):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                counts[i] += 1
    counts = list(accumulate(counts))
    ax0.plot(ave_grads, alpha=0.3, color="b")
    ax0.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k" )
    plt.xticks(counts[:-1], ['interface' for i in counts[:-1]])
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient Magnitude")
    plt.title("Gradient flow")
    plt.grid(True)
    fig0.savefig(out_path)
    plt.close(fig0)