from utils.visualization import *
import numpy as np
import random

# Debugging tool
from IPython import embed

if __name__ == '__main__':
    print("Inside test_visualization")

    #Generate random array vals for frame_id and track_id
    frame_id = np.random.randint(10, size=80)
    track_id = np.random.randint(30, size=80)

    # Fake Data and called the graph visualization function
    y_out = np.vstack((frame_id, track_id)).T
    y_in = y_out #Equal for the time being...Such is life

    generate_dynamic_graph(y_in, y_out)
