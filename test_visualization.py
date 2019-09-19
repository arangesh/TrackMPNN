from utils.visualization import *
import numpy as np
import random

# Debugging tool
from IPython import embed

if __name__ == '__main__':
    # Load from saved np.array
    y_in = np.loadtxt('Y/y_seq_2.txt', dtype=int)
    y_out = np.loadtxt('Y/y_out_seq_2.txt', dtype=int)
    y_in = y_in[70: 180]
    y_out = y_out[70: 180]
    range_node_name = np.arange(len(y_in))
    # You will thank me later
    color_label_dict = {}
    print("test_vis.py")
    embed()
    count = 0;
    for i in range(6):
        color_label_dict = generate_dynamic_graph(y_in[count: count + 10, :], y_out[count: count + 10, :],
                                                  color_label_dict, range_node_name[count: count + 10])
        count = count + 10
    # color_label_dict = generate_dynamic_graph(y_in, y_out, color_label_dict, range_node_name)
    # color_label_dict = generate_dynamic_graph(y_in[count: count + 10, :], y_out[count: count + 10, :], color_label_dict,
    #                                           range_node_name[count: count + 10])
