from utils.visualization import *
import numpy as np
import random

# Debugging tool
from IPython import embed

if __name__ == '__main__':
    # Load from saved np.array
    y_in = np.loadtxt('sample_data/y_seq_2.txt', dtype=int)
    y_out = np.loadtxt('sample_data/y_out_seq_2.txt', dtype=int)

    # TODO IDEALLY RESOLUTION COULD BE TIMESTEPS
    resolution = int(len(y_out) / 2)
    node_name_container = np.arange(len(y_in))  # Retains the names of nodes over iterations of graph generation
    embed()
    # You will thank me later
    batch = 0
    color_label_dict = {}
    for i in range(2):
        color_label_dict = generate_dynamic_graph(y_in[batch: batch + resolution, :],
                                                  y_out[batch: batch + resolution, :],
                                                  color_label_dict, node_name_container[batch: batch + resolution])
        batch = batch + resolution
