"""
NOTE: Make sure feature matrix X is initialized as a cuda tensor with requires_grad=True outside this module
These functions have been designed to be used in a loop over timesteps.
Normal usage would be something like:

G = initialize_graph()
loss = network.forward(G)
for t = 2:T // because intialization step accounts for t=0 and t=1
    G = update_graph(G, t)
    loss += network.forward(G)
    if condition:
        prune_graph() // helps reduce memory usage and streamline information flow
    G, y_out = decode_tracks(G, y_out) // decode and finalize tracks upto a certain timestep and remove those nodes/edges from graph
end
network.backward(loss)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch


def normalize(adj):
    """Row-normalize sparse matrix (from Kipf and Welling)"""
    rowsum = torch.sum(adj, 1)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0
    r_mat_inv = torch.diag(r_inv)

    return torch.matmul(r_mat_inv, adj)


def hungarian(node_adj, scores, y_pred, t, threshold=0.5):
    """
    This is a function that provides the optimal linear assignment for a set of detections
    from a given timestep using the Hungarian algorithm

    node_adj [N, N]: Adjacency matrix for updating association nodes
    scores [N, 2]: Predicted binary class probabilities for each node
    y_pred [N, 3]: Array where each row is [ts, det_id, ass_id] indicating the associated detection
                   for each node i.e. ith row entry indicates the timestep of current node, detection
                   id of the current node, and the detection id of next associated node
    t [scalar]: Timestep at which to carry out optimal linear assignment
    threshold [scalar]: Max aaceptable value of cost to carry out matching

    Returns: 
    y_pred [N, 3]: Updated y_pred with optimal assignment for time t
    """
    # find detection indices for timestep t
    idx_t = np.where(y_pred[:, 0] == t)[0]
    if idx_t.size == 0:
        return y_pred

    # find indices of all previous edges connected to detections from timestep t
    idx_prev_edges = np.array([], dtype='int64')
    for i in idx_t:
        idx_prev_edges = np.concatenate((idx_prev_edges, np.where(node_adj[:i, i])[0]), 0) # find edge nodes it is connected to
    if idx_prev_edges.size == 0:
        return y_pred

    # find indices of all previous detections connected to detections from timestep t
    idx_prev = np.array([], dtype='int64')
    for i in idx_prev_edges:
        idx_prev = np.concatenate((idx_prev, np.where(node_adj[i, :])[0][0:1]), 0)

    # remove duplicates
    idx_prev = np.unique(idx_prev)
    # remove already associated detections
    idx_prev = idx_prev[np.where(y_pred[idx_prev, 2] == -1)[0]]
    
    # create cost matrix for bipartite matching problem
    C = np.full((idx_prev.size, idx_t.size), 100.0, dtype='float32')
    for i, id_prev in enumerate(idx_prev):
        for j, id_t in enumerate(idx_t):
            set1 = np.where(node_adj[:id_t, id_prev])[0]
            set2 = np.where(node_adj[:id_t, id_t])[0]
            edge_id = np.intersect1d(set1, set2)
            if edge_id.size == 0:
                continue
            elif edge_id.size == 1:
                C[i, j] = scores[edge_id.item(), 0] # use first entry because we need a cost value
            else:
                assert False, "Two detection nodes connected through more than one edge!"

    # find and carry out optimal linear assignment
    row_ind, col_ind = linear_sum_assignment(C)
    for i, j in zip(row_ind, col_ind):
        if C[i, j] > threshold: # if cost is more than threshold, do not associate
            continue
        else:
            y_pred[idx_prev[i], 2] = y_pred[idx_t[j], 1]

    return y_pred


def initialize_graph(X, y, t_st=0, mode='test', cuda=True):
    """
    This is a function for initializing the graph on which to perform
    message passing operations.

    X [B, NUM_DETS, NUM_FEATS]: Features for all detections in a sequence
    y [B, NUM_DETS, 2]: Array where each row is [ts, track_id]

    Returns:
    y_pred [N, 3]: Array where each row is [ts, det_id, ass_id] indicating the associated detection
                   for each node i.e. ith row entry indicates the timestep of current node, detection
                   id of the current node, and the detection id of next associated node
    feats [N, NUM_FEATS]: Input features for each node in graph (includes detection
                          nodes and edge nodes (0s))
    node_adj [N, N]: Adjacency matrix for updating association nodes
    edge_adj [N, N]: Adjacency matrix for updating detection nodes
    labels [N,]: Binary class for each node
    t1+1 scalar: Scalar value indicating next timestep to be processed
    tN+1 scalar: Scalar value indicating the last timestep in the sequence
    """
    # remove dummy batch node (only support batch size 1)
    assert (X.size()[0] == y.size()[0] == 1), "Only batch size 1 supported!"
    assert (X.size()[1] == y.size()[1]), "Input dimension mismatch!"

    # find first two non-empty times on or after t_st to initialize graph
    times = torch.sort(y[0, :, 0])[0]
    for t in times:
        if t >= t_st:
            t0 = t1 = t
            break
    tN = times[-1]
    for t in times:
        if t > t0:
            t1 = t
            break
    t0, t1, tN = int(t0.item()), int(t1.item()), int(tN.item())
    if (t0 == t1) or ((y[0, :, 1] == -1).all() and mode == 'train'):
        return None, None, None, None, None, None, None

    ids_t0 = torch.nonzero(y[0, :, 0] == t0)[:, 0]
    ids_t1 = torch.nonzero(y[0, :, 0] == t1)[:, 0]
    num_dets_t0 = ids_t0.size()[0]
    num_dets_t1 = ids_t1.size()[0]

    # initialize y_pred
    y_pred = -1*torch.ones((num_dets_t0+num_dets_t0*num_dets_t1+num_dets_t1, 3), dtype=torch.int64) # (N0+N1+N0*N1, 3)
    y_pred[:num_dets_t0, 0] = t0
    y_pred[num_dets_t0+num_dets_t0*num_dets_t1:, 0] = t1
    y_pred[:num_dets_t0, 1] = ids_t0
    y_pred[num_dets_t0+num_dets_t0*num_dets_t1:, 1] = ids_t1

    # initialize feats
    X_edge = torch.zeros((num_dets_t0*num_dets_t1, X.size()[2])) # (N0*N1, NUM_FEATS)
    feats = torch.cat((X[0, ids_t0, :], X_edge.to(X.device), X[0, ids_t1, :]), 0) # (N0+N0*N1+N1, NUM_FEATS)

    # initialize node_adj
    node_adj = torch.zeros(feats.size()[0], feats.size()[0])
    for i in range(num_dets_t0):
        node_adj[num_dets_t0+i*num_dets_t1:num_dets_t0+(i+1)*num_dets_t1, i] = 1
    for i in range(num_dets_t1):
        node_adj[num_dets_t0+i:num_dets_t0+num_dets_t0*num_dets_t1:num_dets_t1, num_dets_t0+num_dets_t0*num_dets_t1+i] = -1
    # initialize edge_adj
    edge_adj = torch.t(node_adj) # tranpose node_adj to get edge_adj
    # retain node and edge informations
    I_edge = torch.diag((y_pred[:, 0] == -1).float())
    I_node = torch.eye(y_pred.size()[0]) - I_edge
    node_adj = node_adj + I_node
    edge_adj = edge_adj + I_edge

    # initialize labels
    labels = torch.zeros((feats.size()[0],), dtype=torch.int64) # (1, N0+N0*N1+N1)
    y_t0 = y[0, ids_t0, :] # (N0, 2)
    y_t1 = y[0, ids_t1, :] # (N1, 2)
    labels[:num_dets_t0] = y_t0[:, 1] >= 0
    labels[num_dets_t0+num_dets_t0*num_dets_t1:] = y_t1[:, 1] >= 0
    for i in range(num_dets_t1):
        if y_t1[i, 1] == -1: # if a false positive, no edge is positive
            continue
        idx = torch.nonzero(y_t0[:, 1] == y_t1[i, 1])[:, 0]
        if idx.size()[0] == 1:
            labels[num_dets_t0+idx[0]*num_dets_t1+i] = 1
        elif idx.size()[0] > 1:
            assert False, "More than one detection from same timestep assinged to same track!"

    if cuda:
        y_pred = y_pred.cuda()
        node_adj = node_adj.to_sparse().cuda()
        edge_adj = edge_adj.to_sparse().cuda()
        labels = labels.cuda()

    return y_pred, feats, node_adj, edge_adj, labels, t1+1, tN+1


def update_graph(node_adj, labels, scores, y_pred, X, y, t, use_hungraian=True, mode='test', cuda=True):
    """
    This is a function for updating the graph with detections from timestep t and performing
    other upkeep operations.

    node_adj [N, N]: Adjacency matrix for updating association nodes
    labels [N,]: Binary class for each node
    scores [N, 2]: Predicted binary class probabilities for each node
    y_pred [N, 3]: Array where each row is [ts, det_id, ass_id] indicating the associated detection
                   for each node i.e. ith row entry indicates the timestep of current node, detection
                   id of the current node, and the detection id of next associated node
    X [B, NUM_DETS, NUM_FEATS]: Features for all detections in a sequence
    y [B, NUM_DETS, 2]: Array where each row is [ts, track_id]
    t [scalar]: Timestep whose detections to update the graph with

    Returns: (Typically N' > N)
    y_pred [N', 3]: Updated y_pred
    feats [N'-N, NUM_FEATS]: input feats for new nodes and edges
    node_adj [N', N']: Updated node_adj
    edge_adj [N', N']: Updated edge_adj
    labels [N',]: Updated labels
    """
    # remove dummy batch node (only support batch size 1)
    assert (X.size()[0] == y.size()[0] == 1), "Only batch size 1 supported!"
    assert (X.size()[1] == y.size()[1]), "Input dimension mismatch!"

    # move everything to CPU
    node_adj = node_adj.detach().cpu().to_dense().numpy().astype('float32')
    if labels is not None:
        labels = labels.detach().cpu().numpy().astype('int64')
    scores = scores.detach().cpu().numpy().astype('float32')
    y_pred = y_pred.detach().cpu().numpy().astype('int64')
    y = y.squeeze(0).detach().cpu().numpy().astype('int64')

    # reset diagonals to zeros
    diag_ind = np.diag_indices(node_adj.shape[0])
    node_adj[diag_ind] = 0

    # y_pred_t-1 <-- update(scores_t-1)
    y_pred[:, 2] = -1
    if mode == 'train': # during training, make use of the ground truth to update graph
        for i in range(y_pred.shape[0]):
            if y_pred[i, 0] < 0: # if edge node, continue
                continue
            if labels[i] == 1: # if detection is a true positive
                ids = np.where(node_adj[i+1:, i])[0]+i+1 # find edge nodes it is connected to
                idx = np.where(labels[ids])[0] # find edges with +ve label
                idx = ids[idx] # actual indices of all +ve edges
                if idx.size == 0: # if no positive association exists
                    continue
                elif idx.size == 1: # if positive association exists
                    idx_det = np.where(node_adj[idx.item(), :])[0][-1]
                    y_pred[i, 2] = y_pred[idx_det, 1]
                else:
                    assert False, "More than one GT edge from same node!"
            else: # if detection is false positive
                y_pred[i, 2] = y_pred[i, 1] # self assignment so that it remains inactive
    else: # during inference, make use of previous model predictions to update graph
        if use_hungraian:
            for t_match in range(y_pred[0, 0], y_pred[-1, 0]+1):
                y_pred = hungarian(node_adj, scores, y_pred, t_match)
        else:
            for i in range(y_pred.shape[0]):
                if y_pred[i, 0] < 0: # if edge node, continue
                    continue
                if scores[i, 1] >= 0.5: # if detection is a true positive
                    ids = np.where(node_adj[i+1:, i])[0]+i+1 # find edge nodes it is connected to
                    idx = np.where(scores[ids, 1] >= 0.5)[0] # find edges with +ve label
                    idx = ids[idx] # actual indices of all +ve edges
                    # only retain edges that connect to a true positive detection
                    idx = np.array([x for x in idx if scores[np.where(node_adj[x, :])[0][-1], 1] >= 0.5], dtype='int64')
                    if idx.size > 0: # if positive association exists
                        idx_next_det = np.where(y_pred[:, 0] >= 0)[0] # find all detection nodes
                        idx_next_det = idx_next_det[idx_next_det > idx[0]] # find first detection after first positive edge
                        idx = idx[idx < idx_next_det[0]] # retain edges only from nearest timestep 

                        best_idx = idx[np.argmax(scores[idx, 1])] # assign to edge with highest prob in nearest timestep
                        # get detection index to which highest scoring edge connects, and associate to it
                        best_idx_det = np.where(node_adj[best_idx, :])[0][-1]
                        y_pred[i, 2] = y_pred[best_idx_det, 1]

    num_past = y_pred.shape[0]
    if mode == 'train':
        # find true positive detections that are not yet associated, and make them available for association at time t
        t_prev = np.amax(y_pred[y_pred[:, 0] < t, 0])
        ids_active_pred = np.where(np.logical_or(np.logical_and(y_pred[:, 0] != -1, y_pred[:, 2] == -1), y_pred[:, 0] == t_prev))[0]
    else:
        # find true positive detections that are not yet associated, and make them available for association at time t
        ids_active_pred = np.where(np.logical_and(np.logical_and(y_pred[:, 0] != -1, y_pred[:, 2] == -1), scores[:, 1] >= 0.5))[0]
    num_dets_active = ids_active_pred.size
    ids_t = np.where(y[:, 0] == t)[0]
    num_dets_t = ids_t.size
    pad_size = num_dets_active*num_dets_t+num_dets_t

    # y_pred_t <-- y_pred_t-1
    if num_dets_t != 0: # negative indexing does not work if num_dets_t is 0
        y_pred = np.concatenate((y_pred, -1*np.ones((pad_size, 3), dtype='int64')), 0)
        y_pred[-num_dets_t:, 0] = t
        y_pred[-num_dets_t:, 1] = ids_t
    y_pred = torch.from_numpy(y_pred)

    # feats_t <-- feats_t-1
    X_edge = torch.zeros((num_dets_active*num_dets_t, X.size()[2]))
    feats = torch.cat((X_edge.to(X.device), X[0, ids_t, :]), 0)

    # [node_adj_t, edge_adj_t] <-- node_adj_t-1
    if num_dets_t != 0: 
        node_adj = np.concatenate((node_adj, np.zeros((pad_size, node_adj.shape[1]), dtype='float32')), 0)
        node_adj = np.concatenate((node_adj, np.zeros((node_adj.shape[0], pad_size), dtype='float32')), 1)
        for i in range(num_dets_active):
            node_adj[num_past+i*num_dets_t:num_past+(i+1)*num_dets_t, ids_active_pred[i]] = 1
        for i in range(num_dets_t):
            node_adj[num_past+i:num_past+num_dets_active*num_dets_t:num_dets_t, num_past+num_dets_active*num_dets_t+i] = -1
    node_adj = torch.from_numpy(node_adj).to_sparse()
    edge_adj = node_adj.transpose(1, 0) # tranpose node_adj to get edge_adj
    # retain node and edge informations
    I_edge = torch.diag((y_pred[:, 0] == -1).float()).to_sparse()
    I_node = torch.eye(y_pred.size()[0]).to_sparse() - I_edge
    node_adj = node_adj + I_node
    edge_adj = edge_adj + I_edge

    # labels_t <-- labels_t-1
    if labels is not None:
        if num_dets_t != 0:
            labels = np.concatenate((labels, np.zeros((pad_size,), dtype='int64')), 0)
            y_active = y[y_pred[ids_active_pred, 1].detach().cpu().numpy().astype('int64'), :]
            y_t = y[ids_t, :]
            labels[num_past+num_dets_active*num_dets_t:] = y_t[:, 1] >= 0
            if y_t.size > 0:
                for i in range(num_dets_t):
                    # if a false positive, no edge is positive (this may not be needed because all active dets are true positives)
                    if y_t[i, 1] == -1:
                        continue
                    idx = np.where(y_active[:, 1] == y_t[i, 1])[0]
                    for ids in idx: # for each det from the same track
                        labels[num_past+ids*num_dets_t+i] = 1
        labels = torch.from_numpy(labels)
        if cuda:
            labels = labels.cuda()

    if cuda:
        y_pred = y_pred.cuda()
        node_adj = node_adj.cuda()
        edge_adj = edge_adj.cuda() 

    return y_pred, feats, node_adj, edge_adj, labels


def prune_graph(states, node_adj, labels, scores, y_pred, t_st, t_ed, threshold=0.5, cuda=True):
    """
    This is a function for pruning low probability nodes and edges.

    states [N, NUM_HIDDEN]: Hidden states for each node in graph
                          nodes and edge nodes (0s))
    node_adj [N, N]: Adjacency matrix for updating association nodes
    labels [N,]: Binary class for each node
    scores [N, 2]: Predicted binary class probabilities for each node
    y_pred [N, 3]: Array where each row is [ts, det_id, ass_id] indicating the associated detection
                   for each node i.e. ith row entry indicates the timestep of current node, detection
                   id of the current node, and the detection id of next associated node
    t_st, t_ed [scalars]: Start and end timesteps for pruning (inclusive)
    threshold [scalars]: Threshold for pruning edges

    Returns: (Typically N' < N)
    y_pred [N', 3]: Pruned y_pred
    states [N', NUM_HIDDEN]: Pruned states
    node_adj [N', N']: Pruned node_adj
    labels [N',]: Pruned labels
    scores [N', 2]: Pruned scores
    """
    assert (t_st <= t_ed), "t_st must be lesser than or equal to t_ed!"
    
    idx = torch.nonzero((y_pred[:, 0] >= t_st) & (y_pred[:, 0] <= t_ed))[:, 0]
    if idx.size()[0] == 0:
        return y_pred, states, node_adj, labels, scores
    idx_st = idx[0]
    idx_ed = idx[-1]

    indices = torch.arange(y_pred.size()[0])
    if cuda:
        indices = indices.cuda()

    # retain graph node if at least one of these conditions is satisfied:
    # --> node has probability greater than threshold
    # --> node belongs to a detection
    # --> node belongs to a timestep before t_st
    # --> node belongs to a timestep after t_ed
    idx = torch.nonzero((scores[:, 1] >= threshold) | (y_pred[:, 0] != -1) | 
        (indices < idx_st) | (indices > idx_ed))[:, 0]

    y_pred   = y_pred[idx, :]
    states    = states[idx, :]
    node_adj = node_adj.to_dense()
    node_adj = node_adj[idx, :]
    node_adj = node_adj[:, idx]
    node_adj = node_adj.to_sparse()
    if labels is not None:
        labels = labels[idx]
    scores   = scores[idx, :]

    return y_pred, states, node_adj, labels, scores


def decode_tracks(states, node_adj, labels, scores, y_pred, y_out, t_upto, ret_win_size, use_hungraian=True, cuda=True):
    """
    This is a function for decoding and finalizing tracks for early parts of the graph,
    and removing said parts from the graph for all future operations.
    This function facilitates the use of the tracker on a rolling window basis, 
    where the timesteps that are processed keep moving forward with a fixed window size,
    while carrying forward tracks from the past.

    states [N, NUM_HIDDEN]: Hidden states for each node in graph
    node_adj [N, N]: Adjacency matrix for updating association nodes
    labels [N,]: Binary class for each node
    scores [N, 2]: Predicted binary class probabilities for each node
    y_pred [N, 3]: Array where each row is [ts, det_id, ass_id] indicating the associated detection
                   for each node i.e. ith row entry indicates the timestep of current node, detection
                   id of the current node, and the detection id of next associated node
    y_out [NUM_DETS, 2]: Array of past tracks where each row is [ts, track_id]
    t_upto [scalar]: Timestep upto which (not including) tracks are to be decoded and then removed from the graph
    ret_win_size [scalar]: Timesteps before t_upto after which detections classified as TP are to be retained for association

    Returns: (Typically N' < N)
    y_pred [N', 3]: Updated y_pred
    y_out [NUM_DETS, 2]: Updated y_out
    states [N', NUM_HIDDEN]: Updated states
    node_adj [N', N']: Updated node_adj
    labels [N',]: Updated labels
    scores [N', 2]: Updated scores
    """
    # move everything to CPU
    states = states.detach().cpu().numpy().astype('float32')
    node_adj = node_adj.detach().cpu().to_dense().numpy().astype('float32')
    if labels is not None:
        labels = labels.detach().cpu().numpy().astype('int64')
    scores = scores.detach().cpu().numpy().astype('float32')
    y_pred = y_pred.detach().cpu().numpy().astype('int64')

    # reset diagonals to zeros
    diag_ind = np.diag_indices(node_adj.shape[0])
    node_adj[diag_ind] = 0

    # y_pred_t-1 <-- update(scores_t-1)
    y_pred[:, 2] = -1
    if use_hungraian:
        for t_match in range(y_pred[0, 0], y_pred[-1, 0]+1):
            y_pred = hungarian(node_adj, scores, y_pred, t_match)
    else:
        for i in range(y_pred.shape[0]):
            if y_pred[i, 0] < 0: # if edge node, continue
                continue
            if scores[i, 1] >= 0.5: # if detection is a true positive
                ids = np.where(node_adj[i+1:, i])[0]+i+1 # find edge nodes it is connected to
                idx = np.where(scores[ids, 1] >= 0.5)[0] # find edges with +ve label
                idx = ids[idx] # actual indices of all +ve edges
                # only retain edges that connect to a true positive detection
                idx = np.array([x for x in idx if scores[np.where(node_adj[x, :])[0][-1], 1] >= 0.5], dtype='int64')
                if idx.size > 0: # if positive association exists
                    idx_next_det = np.where(y_pred[:, 0] >= 0)[0] # find all detection nodes
                    idx_next_det = idx_next_det[idx_next_det > idx[0]] # find first detection after first positive edge
                    idx = idx[idx < idx_next_det[0]] # retain edges only from nearest timestep 

                    best_idx = idx[np.argmax(scores[idx, 1])] # assign to edge with highest prob in nearest timestep
                    # get detection index to which highest scoring edge connects, and associate to it
                    best_idx_det = np.where(node_adj[best_idx, :])[0][-1]
                    y_pred[i, 2] = y_pred[best_idx_det, 1]

    # update and decide on (finalize) tracking predictions upto timestep t_upto
    next_track_id = np.amax(y_out[:, 1])+1 # track id for next new track
    visited = np.zeros((y_out.shape[0],), dtype='int64') # keep track of visited detection nodes
    for i in range(y_out.shape[0]):
        det_id = i
        node_id = np.where(y_pred[:, 1] == det_id)[0]
        # if track has already been finalized, continue
        if node_id.size == 0:
            visited[det_id] = 1
            continue
        # do not start a track from a node after t_upto, or from a node that's a false positive
        if (y_pred[node_id, 0] >= t_upto) or (scores[node_id, 1] < 0.5):
            visited[det_id] = 1
            continue
        # do not start a track from a node that's already been visited
        if visited[det_id]:
            continue

        # figure out if this is the start of a new track or continuation of existing track
        if y_out[det_id, 1] == -1:
            cur_track_id = next_track_id
            next_track_id += 1
        else:
            cur_track_id = y_out[det_id, 1]
        # start accumulating tracks from current detection node
        while True:
            visited[det_id] = 1 # mark node as visited
            y_out[det_id, 1] = cur_track_id # add node to track
            if y_pred[node_id, 2] == -1: # if end of track, exit
                break
            # exit if successive nodes are at or after t_upto
            if y_out[det_id, 0] >= t_upto and y_out[y_pred[node_id, 2], 0] >= t_upto:
                break
            det_id = y_pred[node_id, 2] # move to next node in track
            node_id = np.where(y_pred[:, 1] == det_id)[0]

    # delete parts of graph upto timestep t_upto
    max_id = np.where(np.logical_and(y_pred[:, 0] < t_upto, y_pred[:, 0] != -1))[0] + 1
    if max_id.size == 0:
        max_id = 0
    else:
        max_id = max_id[-1] # get index of last detection before t_upto
    del_ids = np.arange(max_id, dtype='int64')
    # retain unassociated TP dets, and for other dets-remove edges to dets after and at t_upto
    retain_ids = np.array([], dtype='int64')
    for idx in range(max_id):
        if y_pred[idx, 0] == -1: # if it is an edge node
            continue
        else: # if it is a detection node
            if (y_pred[idx, 2] == -1) and (scores[idx, 1] >= 0.5) and (y_pred[idx, 0] >= t_upto - ret_win_size): # if TP and unassociated and within retain window
                retain_ids = np.append(retain_ids, idx) # store id to retain this node
            else:
                # find and remove all edges connected to detection nodes after and at t_upto
                idx_edges = np.where(node_adj[:, idx])[0]
                idx_edges = idx_edges[idx_edges >= max_id]
                del_ids = np.concatenate((del_ids, idx_edges), 0)
    del_ids = np.delete(del_ids, retain_ids, 0) # remove ids to be retained from those to be deleted

    states = states[np.delete(np.arange(y_pred.shape[0]), del_ids, 0), :]
    node_adj = np.delete(node_adj, del_ids, 0)
    node_adj = np.delete(node_adj, del_ids, 1)
    if labels is not None:
        labels = np.delete(labels, del_ids, 0)
    scores = np.delete(scores, del_ids, 0)
    y_pred = np.delete(y_pred, del_ids, 0)

    # take everything back to GPU
    y_pred = torch.from_numpy(y_pred)
    states = torch.from_numpy(states)
    node_adj = torch.from_numpy(node_adj).to_sparse()
    I_edge = torch.diag((y_pred[:, 0] == -1).float()).to_sparse()
    I_node = torch.eye(y_pred.size()[0]).to_sparse() - I_edge
    node_adj = node_adj + I_node
    labels = torch.from_numpy(labels)
    scores = torch.from_numpy(scores)

    if cuda:
        y_pred = y_pred.cuda()
        states = states.cuda()
        node_adj = node_adj.cuda()
        labels = labels.cuda()
        scores = scores.cuda()

    return y_pred, y_out, states, node_adj, labels, scores
