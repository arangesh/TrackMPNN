# This script contains util functions that help in different visualizations of TrackMPNN codebase
# Work In Progress. Enjoy your long weekend!

import numpy as np
import networkx as nx
from random import randrange
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite

# Debugging tool
from IPython import embed


def generate_track_association(y_in, y_out):
    '''
    :param y_in:  An np array of size [N, 2] where N is the no. of input detections
    :param y_out: An np array of size [N, 2] , same as the length of y_in
    :return: list of detections belong to the same track

    #RIGHT NOW IT ASSOCIATES ONLY IF ALL TRACK IDS MATCH
    '''

    # Find the set of track ID's
    unique_track_IDs = set(y_out[:, 1])
    print("Track IDs are: ", unique_track_IDs)

    # Initialize a list to contain info about track associations over different frames
    track_keeping = []

    # Compare and assert the track association of y_out with y_in
    for id in unique_track_IDs:
        det_idx = np.where(y_out[:, 1] == id)[0]
        print("curr_id: ", id, " corrensponding dets det_idx: ", det_idx)

        # Find correct track associations and create edges
        if (y_in[det_idx][:, 1] == y_out[det_idx][:, 1]).all():
            seq_edges = zip(list(det_idx), list(det_idx[1:]))
            track_keeping.append(seq_edges)

    return track_keeping


def generate_dynamic_graph(y_in, y_out):
    """
    This function dynamically generates bipartite graphs when an np.array of input and output detections are given
    :y_in:  An np array of size [N, 2] where N is the no. of  detections
    :y_out: An np array of size [N, 2] where N is the no. of  detections
    :return: None; Just visual rendering of the graph
    """

    # Greate your Bipartite graph object
    G = nx.Graph()
    # Extract unique frame ID's
    unique_frames = set(y_out[:, 0])

    pos = {}  # Initialize variable to hold groups of nodes
    label_dict = {}  # label the nodes yo!
    counter = 0  # Just to maintain sanity
    track_keeping = generate_track_association(y_in, y_out)  # Generate association for maintaining the track
    # color_map = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]
    color_map = ["b", "g", "r", "c", "m", "y", "b", "w"]
    color_label = np.asarray(["w" for _ in range(len(y_out))])
    color_label = color_label.astype('<U5')

    for i in unique_frames:
        # find indices that belong to the same frame
        idx = np.where(y_out[:, 0] == i)
        print()
        print("Comparing frame no: ", i, "   ,Idx : ", idx)

        # Add nodes to bipartite set
        curr_nodes = np.asarray(idx)[0]
        G.add_nodes_from(list(curr_nodes), bipartite=i)

        # Update the position of the bipartite set
        counter += 1
        pos.update((node, (counter, index)) for index, node in enumerate(set(curr_nodes.flatten())))

    # Update the node labels
    for idx, node in enumerate(G.nodes()):
        label_dict[node] = node

    # Update the color map
    print("Updating the color map")
    for edge_list in track_keeping:
        for n1, n2 in edge_list:
            print(n1, " connect to ", n2)
            G.add_edge(n1, n2)  # Connect the edges
            track_color = color_map[randrange(len(color_map))]  # Color

            # Associate the nodes with the same color
            color_label[n1] = track_color
            color_label[n2] = track_color

    embed()
    # Draw and Visualize the graph
    plt.subplot(121)
    nx.draw(G, pos=pos, labels=label_dict, node_color=color_label)
    plt.show()
