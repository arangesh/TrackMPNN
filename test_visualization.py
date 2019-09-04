from utils.visualization import *
import numpy as np
import random

# Debugging tool
from IPython import embed

if __name__ == '__main__':
    # Generate random array vals for frame_id and track_id
    frame_id = np.random.randint(7, size=30)
    track_id = np.random.randint(15, size=30)

    # Fake Data and called the graph visualization function
    # y_out = np.vstack((frame_id, track_id)).T
    # y_in = y_out + 1 #Equal for the time being
    # y_in[:,0] =y_out[:,0] #Setting the frame no.s to be equal in y_in and y_out

    #Load from saved np.array
    y_in = np.loadtxt('Y/y_seq_2.txt', dtype=int)
    y_out = np.loadtxt('Y/y_out_seq_2.txt', dtype=int)
    y_in = y_in[100: 165]
    y_out = y_out[100: 165]

    # embed()
    generate_dynamic_graph(y_in, y_out)
