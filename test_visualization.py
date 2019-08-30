from utils.visualization import *
import numpy as np
import random

# Debugging tool
from IPython import embed

if __name__ == '__main__':
    #Generate random array vals for frame_id and track_id
    frame_id = np.random.randint(7, size=30)
    track_id = np.random.randint(15, size=30)

    # Fake Data and called the graph visualization function
    y_out = np.vstack((frame_id, track_id)).T
    y_in = y_out + 1 #Equal for the time being
    y_in[:,0] =y_out[:,0] #Setting the frame no.s to be equal in y_in and y_out

    generate_dynamic_graph(y_in, y_out)
