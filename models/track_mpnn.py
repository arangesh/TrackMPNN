import torch
import torch.nn as nn
import torch.sparse as sp

from models.layers import FactorGraphGRU


class TrackMPNN(nn.Module):
    def __init__(self, nfeatures, nhidden, nattheads, msg_type):
        super(TrackMPNN, self).__init__()
        self.input_transform_1 = nn.Linear(nfeatures, nhidden, bias=True)
        self.input_transform_1.weight.data.normal_(mean=0.0, std=0.01)
        self.input_transform_1.bias.data.uniform_(0, 0)
        self.input_transform_2 = nn.Linear(nhidden, nhidden, bias=True)
        self.input_transform_2.weight.data.normal_(mean=0.0, std=0.01)
        self.input_transform_2.bias.data.uniform_(0, 0)
        self.input_transform = nn.Sequential(self.input_transform_1, nn.BatchNorm1d(nhidden), nn.ReLU(), self.input_transform_2)

        self.factor_gru1 = FactorGraphGRU(nhidden, nattheads, msg_type, True)

        self.output_transform_node = nn.Linear(nhidden, 1, bias=True)
        self.output_transform_node.weight.data.normal_(mean=0.0, std=0.01)
        self.output_transform_node.bias.data.uniform_(+4.595, +4.595)

        self.output_transform_edge = nn.Linear(nhidden, 1, bias=True)
        self.output_transform_edge.weight.data.normal_(mean=0.0, std=0.01)
        self.output_transform_edge.bias.data.uniform_(-4.595, -4.595)

        self.output_activation = nn.Sigmoid()

    def forward(self, x, h_in, node_adj, edge_adj):
        I_node = torch.diag(torch.diag(node_adj.to_dense())).to_sparse() # (N', N')
        I_edge = torch.diag(torch.diag(edge_adj.to_dense())).to_sparse() # (N', N')

        if x.size()[0] > 0:
            x = self.input_transform(x) # (N'-N, nhidden)

            h_update = sp.mm(I_node.to_dense()[-x.size()[0]:, -x.size()[0]:].to_sparse(), x) # (N'-N, nhidden)
            if h_in is None: # (N, nhidden)
                h = h_update # (N', nhidden)
            else:
                h = torch.cat((h_in, h_update), dim=0) # (N', nhidden)
        else:
            h = h_in

        h_out = self.factor_gru1(h, node_adj, edge_adj) # (N', nhidden)
        y = sp.mm(I_node, self.output_transform_node(h_out)) + sp.mm(I_edge, self.output_transform_edge(h_out)) # (N', 1)

        return self.output_activation(y), y, h_out
