import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_targets(labels, node_adj, idx_node):
    idx_node = idx_node.detach().cpu().numpy().astype('int64') # (D,)
    targets = torch.zeros_like(labels) # (N,)
    targets[idx_node] = labels[idx_node] # replicate labels for detection nodes
    labels = labels.view(1, -1) # (1, N)
    node_adj = node_adj.detach().cpu().to_dense().numpy().astype('int64') # (N, N)
    # reset diagonals to zeros
    diag_ind = np.diag_indices(node_adj.shape[0])
    node_adj[diag_ind] = 0

    for idx in idx_node:
        # for edges from the past
        idx_ce = np.nonzero(node_adj[:idx, idx])[0]
        if idx_ce.size > 0:
            # find edges connected to detections from same track
            pos_edges = torch.nonzero((labels[:, idx_ce]))[:, 1]
            if pos_edges.numel() == 0: # if no positive edge
                pass
            elif pos_edges.numel() == 1: # if only one positive edge
                targets[idx_ce[pos_edges]] = 1
            elif pos_edges.numel() > 1: # if multiple positive edges
                # use the edge connected to the latest positive detection
                targets[idx_ce[pos_edges[-1:]]] = 1

        # for edges to the future
        idx_ce = idx + np.nonzero(node_adj[idx:, idx])[0]
        if idx_ce.size > 0:
            # find edges connected to detections from same track
            pos_edges = torch.nonzero((labels[:, idx_ce]))[:, 1]
            if pos_edges.numel() == 0: # if no positive edge
                pass
            elif pos_edges.numel() == 1: # if only one positive edge
                targets[idx_ce[pos_edges]] = 1
            elif pos_edges.numel() > 1: # if multiple positive edges
                # use the edge connected to the earliest positive detection
                targets[idx_ce[pos_edges[:1]]] = 1
    return targets


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.eps = 1e-10

    def forward(self, outputs, targets):
        outputs = torch.stack((1 - outputs, outputs), dim=1) # (N, 2)
        targets = targets.view(-1, 1) # (N, 1)

        logpt = torch.log(outputs + self.eps)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            if self.alpha.type() != outputs.data.type():
                self.alpha = self.alpha.type_as(outputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.nll = nn.NLLLoss()

    def forward(self, outputs, targets, node_adj, idx_node):
        idx_node = idx_node.detach().cpu().numpy().astype('int64') # (D,)
        outputs = outputs.view(1, -1) # (1, N)
        targets = targets.view(1, -1) # (1, N)
        node_adj = node_adj.detach().cpu().to_dense().numpy().astype('int64') # (N, N)
        # reset diagonals to zeros
        diag_ind = np.diag_indices(node_adj.shape[0])
        node_adj[diag_ind] = 0

        loss = torch.tensor(0.0).to(outputs.device)
        for idx in idx_node:
            # for edges from the past
            idx_ce = np.nonzero(node_adj[:idx, idx])[0]
            if idx_ce.size > 0:
                # find edges connected to detections from same track
                pos_edges = torch.nonzero((targets[:, idx_ce]))[:, 1]
                if pos_edges.numel() == 0: # if no positive edge
                    pass
                elif pos_edges.numel() == 1: # if only one positive edge
                    loss += F.cross_entropy(outputs[:, idx_ce], pos_edges) / idx_ce.size

            # for edges to the future
            idx_ce = idx + np.nonzero(node_adj[idx:, idx])[0]
            if idx_ce.size > 0:
                # find edges connected to detections from same track
                pos_edges = torch.nonzero((targets[:, idx_ce]))[:, 1]
                if pos_edges.numel() == 0: # if no positive edge
                    pass
                elif pos_edges.numel() == 1: # if only one positive edge
                    loss += F.cross_entropy(outputs[:, idx_ce], pos_edges) / idx_ce.size
        return loss


class EmbeddingLoss(nn.Module):
    def __init__(self, delta_var=0.5, delta_dist=3.0):
        super(EmbeddingLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist

    def forward(self, features, labels):
        """
        features(N, F): Torch tensor with image features corresponding to each detection
        labels(N, 2): Numpy array where each row corresponds to [fr, track_id] for the detection
        """
        cluster_means = []
        # do not consider false positives
        tp_ids = labels[:, 1] >= 0
        features = features[tp_ids, :]
        labels = labels[tp_ids, :]

        # find unique clusters
        cluster_ids = np.unique(labels[:, 1]).tolist()
        C = len(cluster_ids)

        # calculate variance term
        var_loss = torch.tensor(0.0).to(features.device)
        if C > 0:
            for c_id in cluster_ids:
                cluster_means.append(torch.mean(features[labels[:, 1] == c_id, :], dim=0, keepdim=True))
                var_dist = torch.norm(features[labels[:, 1] == c_id, :] - cluster_means[-1], dim=1)
                var_loss += torch.mean(torch.pow(F.relu(var_dist - self.delta_var), 2))
            var_loss /= C

        # calculate distance term
        dist_loss = torch.tensor(0.0).to(features.device)
        if C > 1: # only if multiple clusters exist
            for i in range(C):
                for j in range(C):
                    if i == j:
                        continue
                    mean_dist = torch.norm(cluster_means[i] - cluster_means[j], dim=1)
                    dist_loss += torch.pow(F.relu(self.delta_dist - mean_dist[0]), 2)
            dist_loss /= (C * (C - 1))

        return var_loss + dist_loss
