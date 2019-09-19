from utils.visualization import *
import numpy as np
import random

# Debugging tool
from IPython import embed

if __name__ == '__main__':

    #Load from saved np.array
    y_in = np.loadtxt('Y/y_seq_2.txt', dtype=int)
    y_out = np.loadtxt('Y/y_out_seq_2.txt', dtype=int)
    y_in = y_in[110: 180]
    y_out = y_out[110: 180]

    #You will thank me later
    color_label_dict = {}
    # embed()

    color_label_dict = generate_dynamic_graph(y_in, y_out, color_label_dict)
