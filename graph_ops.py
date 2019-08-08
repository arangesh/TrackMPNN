"""
NOTE: Make sure feature matrix X is initialized as a cuda tensor variable with requires_grad=True outside this module
These functions have been designed to be used in a loop over timesteps.
Normal usage would be something like:

G = initialize_graph()
loss = network.forward(G)
for t = 2:T // because intialization step accounts for t=0 and t=1
    G = update_graph(G, t)
    loss += network.forward(G)
    if condition:
        prune_graph() // helps reduce memory usage and streamline information flow
end
network.backward(loss)
decode_tracks(G)

Note that decode_tracks() can also be called at any intermediate timestep
after the network.forward() has been called.
"""
import torch
import numpy as np
from torch.autograd import Variable


def normalize(adj):
    """Row-normalize sparse matrix (from Kipf and Welling)"""
    rowsum = torch.sum(adj, 1)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0
    r_mat_inv = torch.diag(r_inv)

    return torch.matmul(r_mat_inv, adj)


def initialize_graph(X, y, mode='test', cuda=True):
    """
    This is a function for initializing the graph on which to perform
    graph convolutional operations

    X [B, NUM_DETS, NUM_FEATS]: Features for all detections in a sequence
    y [B, NUM_DETS, 2]: Array where each row is [ts, track_id]

    Returns:
    y_pred [N, 2]: Array where each row is [ts, ass_id] indicating the associated detection
                   for each node i.e. ith row entry indicates the timestep of current node
                   and id of next associated detection node
    feats [N, NUM_FEATS]: Input features for each node in graph (includes detection
                          nodes and edge nodes (0s))
    node_adj [N, N]: Adjacency matrix for updating detection
    edge_adj [N, N]: Adjacency matrix for updating edge nodes
    labels [N,]: Binary class for each node
    t2+1 scalar: Scalar value indicating next timestep to be processed
    tN+1 scalar: Scalar value indicating the last timestep in the sequence
    """
    # remove dummy batch node (only support batch size 1)
    assert (X.size()[0] == y.size()[0] == 1), "Only batch size 1 supported!"
    assert (X.size()[1] == y.size()[1]), "Input dimension mismatch!"

    # find first two non-empty times to initialize graph
    times = torch.sort(y[0, :, 0])[0]
    t1 = t2 = times[0]
    tN = times[-1]
    for t in times:
        if t != t1:
            t2 = t
            break
    if t1 == t2:
        return None, None, None, None, None, None, None


    ids_0 = torch.nonzero(y[0, :, 0] == t1)[:, 0]
    ids_1 = torch.nonzero(y[0, :, 0] == t2)[:, 0]
    num_dets_0 = ids_0.size()[0]
    num_dets_1 = ids_1.size()[0]

    # initialize y_pred
    y_pred = -1*torch.ones((num_dets_0+num_dets_0*num_dets_1+num_dets_1, 2), dtype=torch.int64) # (N0+N1+N0*N1, 2)
    if cuda:
        y_pred = y_pred.cuda()
    y_pred[:num_dets_0, 0] = t1
    y_pred[num_dets_0+num_dets_0*num_dets_1:, 0] = t2

    # initialize feats
    X_edge = torch.zeros((num_dets_0*num_dets_1, X.size()[2])) # (N0*N1, NUM_FEATS)
    if cuda:
        X_edge = X_edge.cuda()
    feats = torch.cat((X[0, ids_0, :], X_edge, X[0, ids_1, :]), 0) # (N0+N0*N1+N1, NUM_FEATS)

    # initialize node_adj
    node_adj = torch.zeros(feats.size()[0], feats.size()[0])
    if cuda:
        node_adj = node_adj.cuda()
    for i in range(num_dets_0):
        node_adj[num_dets_0+i*num_dets_1:num_dets_0+(i+1)*num_dets_1, i] = 1
    for i in range(num_dets_1):
        node_adj[num_dets_0+i:num_dets_0+num_dets_0*num_dets_1:num_dets_1, num_dets_0+num_dets_0*num_dets_1+i] = -1
    # initialize edge_adj
    edge_adj = torch.t(node_adj) # tranpose node_adj to get edge_adj
    # retain node and edge informations
    I_edge = torch.diag(y_pred[:, 0] == -1).float()
    if cuda:
        I_edge = I_edge.cuda()
        I_node = torch.eye(y_pred.size()[0]).cuda() - I_edge
    else:
        I_node = torch.eye(y_pred.size()[0]) - I_edge
    node_adj = node_adj + I_node
    edge_adj = edge_adj + I_edge

    # initialize labels
    if mode == 'train':
        labels = torch.zeros((feats.size()[0],), dtype=torch.int64) # (1, N0+N0*N1+N1)
        if cuda:
            labels = labels.cuda()
        y_0 = y[0, ids_0, :] # (N0, 2)
        y_1 = y[0, ids_1, :] # (N1, 2)
        labels[:num_dets_0] = y_0[:, 1] >= 0
        labels[num_dets_0+num_dets_0*num_dets_1:] = y_1[:, 1] >= 0
        for i in range(num_dets_0):
            if y_0[i, 1] == -1: # if a false positive, no edge is positive
                continue
            idx = torch.nonzero(y_1[:, 1] == y_0[i, 1])[:, 0]
            if idx.size()[0] == 1:
                labels[num_dets_0+i*num_dets_1+idx[0]] = 1
    else:
        labels = None

    # intialize GPU Variable tensors
    # Note that feat is the only output that has a gradient
    node_adj = Variable(node_adj, requires_grad=False)
    edge_adj = Variable(edge_adj, requires_grad=False)
    if labels is not None:
        labels = Variable(labels, requires_grad=False)

    return y_pred, feats, node_adj, edge_adj, labels, t2+1, tN+1


def prune_graph(feats, node_adj, labels_pred, y_pred, t1, t2, threshold=0.5):
    """
    This is a function for pruning low probability nodes and edges

    feats [N, NUM_FEATS]: Input features for each node in graph (includes detection
                          nodes and edge nodes (0s))
    node_adj [N, N]: Adjacency matrix for updating detection nodes
    labels_pred [N, 2]: Predicted binary class probabilities for each node
    y_pred [N, 2]: Array where each row is [ts, ass_id] indicating the associated detection
                   for each node i.e. ith row entry indicates the timestep of current node
                   and id of next associated detection node
    t1, t2 [scalars]: Start and end timesteps for pruning (inclusive)
    threshold [scalars]: Threshold for pruning edges

    Returns: (Typically N' < N)
    y_pred [N', 2]: Pruned y_pred
    feats [N', NUM_FEATS]: Pruned feats
    node_adj [N', N']: Pruned node_adj
    """
    assert (t1 <= t2), "t1 must be lesser than or equal to t2!"
    
    idx = torch.nonzero((y_pred[:, 0] >= t1) & (y_pred[:, 0] <= t2))[:, 0]
    if idx.size()[0] == 0:
        return y_pred, feats, node_adj
    idx_st = idx[0]
    idx_ed = idx[-1]

    # retain graph node if at least one of these conditions is satisfied:
    # --> node has probability greater than threshold
    # --> node belongs to a detection
    # --> node belongs to a timestep before t1
    # --> node belongs to a timestep after t2
    idx = torch.nonzero((labels_pred[:, 1] >= threshold) | (y_pred[:, 0] != -1) | 
        (torch.arange(y_pred.size()[0]) < idx_st) | (torch.arange(y_pred.size()[0]) > idx_ed))[:, 0]

    y_pred   = y_pred[idx, :]
    feats    = feats[idx, :]
    node_adj = node_adj[idx, :]
    node_adj = node_adj[:, idx]

    return y_pred, feats, node_adj


def update_graph(feats, node_adj, labels, labels_pred, y_pred, X, y, t, mode='test', cuda=True):
    """
    This is a function for updating the graph with detections from timestep t

    feats [N, NUM_FEATS]: Input features for each node in graph (includes detection
                          nodes and edge nodes (0s))
    node_adj [N, N]: Adjacency matrix for updating detection nodes
    labels [N,]: Binary class for each node
    labels_pred [N, 2]: Predicted binary class probabilities for each node
    y_pred [N, 2]: Array where each row is [ts, ass_id] indicating the associated detection
                   for each node i.e. ith row entry indicates the timestep of current node
                   and id of next associated detection node
    X [B, NUM_DETS, NUM_FEATS]: Features for all detections in a sequence
    y [B, NUM_DETS, 2]: Array where each row is [ts, track_id]
    t [scalar]: Timestep whose detections to update the graph with

    Returns: (Typically N' > N)
    y_pred [N', 2]: Updated y_pred
    feats [N', NUM_FEATS]: Updated feats
    node_adj [N', N']: Updated node_adj
    edge_adj [N', N']: Updated edge
    labels [N',]: Updated labels
    """
    # remove dummy batch node (only support batch size 1)
    assert (X.size()[0] == y.size()[0] == 1), "Only batch size 1 supported!"
    assert (X.size()[1] == y.size()[1]), "Input dimension mismatch!"

    # move everything to CPU
    node_adj = node_adj.detach().cpu().numpy().astype('float32')
    if labels is not None:
        labels = labels.detach().cpu().numpy().astype('int64')
    labels_pred = labels_pred.detach().cpu().numpy().astype('float32')
    y_pred = y_pred.detach().cpu().numpy().astype('int64')
    y = y.squeeze(0).detach().cpu().numpy().astype('int64')

    # reset diagonals to zeros
    diag_ind = np.diag_indices(node_adj.shape[0])
    node_adj[diag_ind] = 0

    # [y_pred_t-1, feats_t-1, node_adj_t-1, labels_t-1] <-- update(labels_pred_t-1)
    id_offset = np.cumsum(y_pred[:, 0] == -1)
    y_pred[:, 1] = -1
    del_ids = np.array([], dtype='int64')
    for i in range(y_pred.shape[0]):
        if y_pred[i, 0] < 0: # if edge node, continue
            continue
        if labels_pred[i, 1] >= 0.5: # if detection is a true positive
            ids = np.where(node_adj[i+1:, i])[0]+i+1 # find edge nodes it is connected to
            idx = np.where(labels_pred[ids, 1] >= 0.5)[0] # find edges with +ve label
            if idx.size > 0: # if association exists
                idx = ids[idx] # actual indices of all +ve edges
                idx_next_det = np.where(y_pred[:, 0] >= 0)[0] # first find detection nodes
                idx_next_det = idx_next_det[idx_next_det > idx[0]] # find first detection after positive edge
                idx = idx[idx < idx_next_det[0]] # retain indices from nearest timestep only

                best_idx = idx[np.argmax(labels_pred[idx, 1])] # assign to edge with highest prob in nearest timestep
                y_pred[i, 1] = np.where(node_adj[best_idx, :])[0][-1] # find detection node at other end of edge
                y_pred[i, 1] = y_pred[i, 1] - id_offset[y_pred[i, 1]] # offset detection index to account for edge nodes

                # remove all edges after nearest timestep with positive edge
                del_ids = np.concatenate((del_ids, ids[ids > idx_next_det[0]]), 0)

    feats = feats[np.delete(np.arange(y_pred.shape[0]), del_ids, 0), :]
    node_adj = np.delete(node_adj, del_ids, 0)
    node_adj = np.delete(node_adj, del_ids, 1)
    if labels is not None:
        labels = np.delete(labels, del_ids, 0)
    y_pred = np.delete(y_pred, del_ids, 0)

    num_dets_past = y_pred.shape[0] 
    ids_active_pred = np.where(np.logical_and(y_pred[:, 0] != -1, y_pred[:, 1] == -1))[0]
    num_dets_active = ids_active_pred.shape[0]
    ids_t = np.where(y[:, 0] == t)[0]
    num_dets_t = ids_t.shape[0]
    pad_size = num_dets_active*num_dets_t+num_dets_t

    # y_pred_t <-- y_pred_t-1
    y_pred = np.concatenate((y_pred, -1*np.ones((pad_size, 2), dtype='int64')), 0)
    if num_dets_t != 0: # negative indexing does not work if num_dets_t is 0
        y_pred[-num_dets_t:, 0] = t
    y_pred = torch.from_numpy(y_pred)
    if cuda:
        y_pred = y_pred.cuda()

    # feats_t <-- feats_t-1
    X_edge = torch.zeros((num_dets_active*num_dets_t, X.size()[2]))
    if cuda:
        X_edge = X_edge.cuda()
    feats = torch.cat((feats, X_edge, X[0, ids_t, :]), 0)

    # [node_adj_t, edge_adj_t] <-- node_adj_t-1
    node_adj = np.concatenate((node_adj, np.zeros((pad_size, node_adj.shape[1]), dtype='float32')), 0)
    node_adj = np.concatenate((node_adj, np.zeros((node_adj.shape[0], pad_size), dtype='float32')), 1)
    for i in range(num_dets_active):
        node_adj[num_dets_past+i*num_dets_t:num_dets_past+(i+1)*num_dets_t, ids_active_pred[i]] = 1
    for i in range(num_dets_t):
        node_adj[num_dets_past+i:num_dets_past+num_dets_active*num_dets_t:num_dets_t, num_dets_past+num_dets_active*num_dets_t+i] = -1
    node_adj = torch.from_numpy(node_adj)
    if cuda:
        node_adj = node_adj.cuda()
    edge_adj = torch.t(node_adj) # tranpose node_adj to get edge_adj
    # retain node and edge informations
    I_edge = torch.diag(y_pred[:, 0] == -1).float()
    if cuda:
        I_edge = I_edge.cuda()
        I_node = torch.eye(y_pred.size()[0]).cuda() - I_edge
    else:
        I_node = torch.eye(y_pred.size()[0]) - I_edge
    node_adj = node_adj + I_node
    edge_adj = edge_adj + I_edge

    # labels_t <-- labels_t-1
    if mode == 'train':
        labels = np.concatenate((labels, np.zeros((pad_size,), dtype='int64')), 0)
        ids_dets = torch.nonzero(y_pred[:, 0] != -1)[:, 0]
        ids_active = torch.nonzero((y_pred[ids_dets, 0] != t) & (y_pred[ids_dets, 1] == -1))[:, 0]
        ids_active = ids_active.detach().cpu().numpy().astype('int64')
        y_active = y[ids_active, :]

        y_t = y[ids_t, :]
        labels[num_dets_past+num_dets_active*num_dets_t:] = y_t[:, 1] >= 0
        if y_t.size > 0:
            for i in range(num_dets_active):
                if y_active[i, 1] == -1: # if a false positive, no edge is positive
                    continue
                idx = np.where(y_t[:, 1] == y_active[i, 1])[0]
                if idx.size == 1:
                    labels[num_dets_past+i*num_dets_t+idx] = 1
        labels = torch.from_numpy(labels)
        if cuda:
            labels = labels.cuda()

    # intialize GPU Variable tensors
    # Note that feat is the only output that has a gradient
    node_adj = Variable(node_adj, requires_grad=False)
    edge_adj = Variable(edge_adj, requires_grad=False)
    if labels is not None:
        labels = Variable(labels, requires_grad=False)

    return y_pred, feats, node_adj, edge_adj, labels


def decode_tracks(node_adj, labels_pred, y_pred):
    """
    This is a function for decoding tracks from network outputs

    node_adj [N, N]: Adjacency matrix for updating detection nodes
    labels_pred [N, 2]: Predicted binary class probabilities for each node
    y_pred [N, 2]: Array where each row is [ts, ass_id] indicating the associated detection
                   for each node i.e. ith row entry indicates the timestep of current node
                   and id of next associated detection node

    Returns: 
    tracks list([N',]): A list of 1D lists of detection IDs for each track.
    """
    # move everything to CPU
    node_adj = node_adj.detach().cpu().numpy().astype('float32')
    labels_pred = labels_pred.detach().cpu().numpy().astype('float32')
    y_pred = y_pred.detach().cpu().numpy().astype('int64')

    # reset diagonals to zeros
    diag_ind = np.diag_indices(node_adj.shape[0])
    node_adj[diag_ind] = 0

    # [y_pred_t-1, node_adj_t-1, labels_t-1] <-- update(labels_pred_t-1)
    id_offset = np.cumsum(y_pred[:, 0] == -1)
    y_pred[:, 1] = -1
    del_ids = np.array([], dtype='int64')
    for i in range(y_pred.shape[0]):
        if y_pred[i, 0] < 0: # if edge node, continue
            continue
        if labels_pred[i, 1] >= 0.5: # if detection is a true positive
            ids = np.where(node_adj[i+1:, i])[0]+i+1 # find edge nodes it is connected to
            idx = np.where(labels_pred[ids, 1] >= 0.5)[0] # find edges with +ve label
            if idx.size > 0: # if association exists
                idx = ids[idx] # actual indices of all +ve edges
                idx_next_det = np.where(y_pred[:, 0] >= 0)[0] # first find detection nodes
                idx_next_det = idx_next_det[idx_next_det > idx[0]] # find first detection after positive edge
                idx = idx[idx < idx_next_det[0]] # retain indices from nearest timestep only

                best_idx = idx[np.argmax(labels_pred[idx, 1])] # assign to edge with highest prob in nearest timestep
                y_pred[i, 1] = np.where(node_adj[best_idx, :])[0][-1] # find detection node at other end of edge
                y_pred[i, 1] = y_pred[i, 1] - id_offset[y_pred[i, 1]] # offset detection index to account for edge nodes

                # remove all edges after nearest timestep with positive edge
                del_ids = np.concatenate((del_ids, ids[ids > idx_next_det[0]]), 0)

    node_adj = np.delete(node_adj, del_ids, 0)
    node_adj = np.delete(node_adj, del_ids, 1)
    y_pred = np.delete(y_pred, del_ids, 0)
    labels_pred = np.delete(labels_pred, del_ids, 0)

    tracks = []
    visited = np.zeros((y_pred.shape[0],), dtype='int64')
    id_offset = np.cumsum(y_pred[:, 0] == -1)
    id_offset_det = id_offset[y_pred[:, 0] != -1]
    for i in range(y_pred.shape[0]):
    	# do not start a track from an edge, a false positive or a node that has already been visited
        if (y_pred[i, 0] == -1) or (labels_pred[i, 1] < 0.5) or visited[i]:
            continue
        track_ids = []
        cur_id = i-id_offset[i]
        track_ids.append(cur_id)
        while True:
            visited[cur_id+id_offset_det[cur_id]] = 1
            if y_pred[cur_id+id_offset_det[cur_id], 1] == -1:
                break
            cur_id = y_pred[cur_id+id_offset_det[cur_id], 1]
            track_ids.append(cur_id)
        tracks.append(track_ids)

    return tracks
