# This script contains util functions that help in different visualizations of TrackMPNN codebase

import numpy as np
import networkx as nx
import matplotlib
from random import shuffle
import matplotlib.pyplot as plt

def generate_track_association(y_in, y_out, node_name_container):
    '''
    :param y_in:  An np array of size [N, 2] where N is the no. of input detections
    :param y_out: An np array of size [N, 2] , same as the length of y_in
    :return: list of detections belong to the same track

    #RIGHT NOW IT ASSOCIATES ONLY IF ALL TRACK IDS MATCH
    '''
    # Find the set of track ID's
    unique_track_IDs = set(y_out[:, 1])
    # Initialize a list to contain info about track associations over different frames
    track_keeping = []

    # Compare and assert the track association of y_out with y_in
    for id in unique_track_IDs:
        det_idx = np.where(y_out[:, 1] == id)[0]

        # Find correct track associations and create edges
        # if y_in's track ID's at indices det_idx matches, their length of set() shoulb be 1
        if (len(set(y_in[det_idx][:, 1])) == 1):
            # seq_edges = zip(list(det_idx), list(det_idx[1:])) #orig
            seq_edges = zip(list(node_name_container[det_idx]), list(node_name_container[det_idx[1:]]))
            track_keeping.append(seq_edges)

    return track_keeping


# Update node labels and frame labels
def update_node_and_frame_labels(G, G_frame_label):

    #Node label is the label that we see in the renderer
    #Frame label indicates the frame number a set of nodes belong to(usually in the bottom row of the graph)
    node_label_dict = {}
    frame_label_dict = {}

    #fill up node_labels
    for idx, node in enumerate(G.nodes()):
        node_label_dict[node] = node

    # fill up range labels
    for idx, node in enumerate(G_frame_label.nodes()):
        frame_label_dict[node] = "f_" + str(idx)

    return node_label_dict, frame_label_dict


def fill_color_labels(color_label_dict, num_tracks):
    # To color code the associated tracks
    color_label = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]))) + [1.0] for x in
                   np.arange(0, 1, 1.0 / num_tracks)]
    shuffle(color_label)

    # Generating a dict for retracing colors throughouot
    for i in range(len(color_label)):
        #Add only new colors
        if i not in color_label_dict:
            color_label_dict[i] = color_label[i]

    return color_label_dict


def generate_dynamic_graph(y_in, y_out, color_label_dict, node_name_container):
    """
    This function dynamically generates bipartite graphs when an np.array of input and output detections are given
    :y_in:  An np array of size [N, 2] where N is the no. of  detections
    :y_out: An np array of size [N, 2] where N is the no. of  detections
    :return: None; Just visual rendering of the graph
    """

    # Greate your Bipartite graph object
    G = nx.Graph()
    G_frame_label = nx.Graph()
    # Extract unique frame ID's
    unique_frames = sorted(set(y_out[:, 0]))

    pos = {}  # hold position information of the nodes
    pos_frame_label = {}  # position information for the frame labels

    counter = 0  # Just to maintain sanity
    track_keeping = generate_track_association(y_in, y_out,
                                               node_name_container)  # Generate association for maintaining the track
    label_list = []  # Meh! TODO Optimize the color coding code. "color coding code" - That's an alliteration
    # Choose the max num of tracks
    num_tracks = np.amax(y_in[:, 1]) + 1

    #Fill in range of colors to color the tracks
    color_label_dict = fill_color_labels(color_label_dict, num_tracks)
    color_label_arr = np.ones((y_out.shape[0], 4))

    for i in unique_frames:
        # find indices that belong to the same frame
        idx = np.where(y_out[:, 0] == i)
        print("\nFor frame no: ", i, "   ,Idx nodes are : ", idx)

        # Add nodes to bipartite set
        curr_nodes = np.asarray(idx[0])
        G.add_nodes_from(list(node_name_container[idx[0]]), bipartite=i)
        G_frame_label.add_node(i, bipartite=i)

        # Update the position of the bipartite set
        counter += 1
        # Find orig
        pos.update(
            (node_name_container[node], (counter, index)) for index, node in enumerate(set(curr_nodes.flatten())))
        pos_frame_label.update((i, (counter, index - 0.3)) for index, node in enumerate(set(np.array((i, i)))))

    # Update the node labels
    node_label_dict, frame_label_dict = update_node_and_frame_labels(G, G_frame_label)

    # TODO Needs a better solution
    for i in node_label_dict:
        label_list.append(i)
    label_array = np.asarray(label_list)

    # Update the color map
    for edge_list in track_keeping:
        # track_color = color_map[randrange(len(color_map))]  # Color
        for n1, n2 in edge_list:
            print(n1, " connecting to ", n2)
            G.add_edge(n1, n2)  # Connect the edges

            # Associate the nodes with the same color
            n1_idx = np.where(label_array == n1)[0][0]
            n2_idx = np.where(label_array == n2)[0][0]

            color_index = y_in[np.where(node_name_container == n1)[0]][0][1]
            track_color = color_label_dict[color_index]  # Color
            color_label_arr[n1_idx] = track_color
            color_label_arr[n2_idx] = track_color

    # Draw and Visualize the graph
    f = plt.figure(figsize=(15, 10))
    ax = f.add_subplot(111)
    plt.title('Time --------------------------->', loc='left', ha='center', va='bottom', x=0.5, y=0, fontsize='medium',
              fontfamily='cursive', fontweight='light', color='B')
    plt.xlabel('categories')
    plt.ylabel('values')

    nx.draw(G, pos=pos, labels=node_label_dict, node_color=color_label_arr)
    nx.draw(G_frame_label, pos=pos_frame_label, labels=frame_label_dict, font_size=9, font_color='b',
            font='Comic Sans MS', node_color='w')
    plt.show()

    return color_label_dict
