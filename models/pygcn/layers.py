import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feats, adj):
        support = torch.mm(feats, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FactorGraphConvolution(Module):
    """
    Similar to GCN, except different weights for nodes and edges (i.e. variables and factors)
    """
    def __init__(self, in_features, out_features, bias=True, msg_type='subtract'):
        super(FactorGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.msg_type = msg_type
        if self.msg_type == 'concat':
            self.node_weight = Parameter(torch.FloatTensor(2*in_features, out_features))
        else:
            self.node_weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.node_bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('node_bias', None)
        self.edge_weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.edge_bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('edge_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.node_weight.size(1))
        self.node_weight.data.uniform_(-stdv, stdv)
        if self.node_bias is not None:
            self.node_bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.edge_weight.size(1))
        self.edge_weight.data.uniform_(-stdv, stdv)
        if self.edge_bias is not None:
            self.edge_bias.data.uniform_(-stdv, stdv)

    def forward(self, feats, node_adj, edge_adj):
        if self.msg_type == 'concat':
            node_support = torch.cat((torch.spmm((node_adj > 0).float(), feats), 
                torch.spmm((node_adj < 0).float(), feats)), dim=1)
        else:
            node_support = torch.spmm(node_adj, feats)
        node_output = torch.mm(node_support, self.node_weight)
        if self.node_bias is not None:
            node_output = node_output + self.node_bias

        edge_support = torch.spmm(edge_adj, feats)
        edge_output = torch.mm(edge_support, self.edge_weight)
        if self.edge_bias is not None:
            edge_output = edge_output + self.edge_bias

        return F.relu(node_output) + edge_output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'