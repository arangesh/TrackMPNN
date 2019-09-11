import math
import torch
import torch.nn as nn


class FactorGraphGRU(nn.Module):
    """
    Similar to GCN, except different GRU cells for nodes and edges (i.e. variables and factors)
    """
    def __init__(self, nhidden, bias=True, msg_type='concat'):
        super(FactorGraphGRU, self).__init__()
        self.nhidden = nhidden
        self.msg_type = msg_type
        self.bias = bias
        if self.msg_type == 'concat':
            self.edge_gru = nn.GRUCell(2*nhidden, nhidden, bias=self.bias)
        else:
            self.edge_gru = nn.GRUCell(nhidden, nhidden, bias=self.bias)
        self.node_gru = nn.GRUCell(nhidden, nhidden, bias=self.bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.edge_gru.weight_ih.data.normal_(mean=0.0, std=0.01)
        self.edge_gru.weight_hh.data.normal_(mean=0.0, std=0.01)
        if self.bias is not None:
            self.edge_gru.bias_ih.data.uniform_(0, 0)
            self.edge_gru.bias_hh.data.uniform_(0, 0)

        self.node_gru.weight_ih.data.normal_(mean=0.0, std=0.01)
        self.node_gru.weight_hh.data.normal_(mean=0.0, std=0.01)
        if self.bias is not None:
            self.node_gru.bias_ih.data.uniform_(0, 0)
            self.node_gru.bias_hh.data.uniform_(0, 0)

    def forward(self, h, node_adj, edge_adj):
        node_adj = node_adj.to_dense()
        edge_adj = edge_adj.to_dense()
        I_node = torch.diag(torch.diag(node_adj))
        I_edge = torch.diag(torch.diag(edge_adj))

        # node_support: concat(h_n1, h_n2)
        if self.msg_type == 'concat':
            node_support = torch.cat((torch.mm(((node_adj - I_node) > 0).float(), h), 
                torch.mm(((node_adj - I_node) < 0).float(), h)), dim=1)
        else:
            node_support = torch.mm(node_adj - I_node, h)
        # node_output: GRU(h_e(t-1), concat(h_n1, h_n2))
        edge_output = self.edge_gru(node_support, h)

        # edge_support: sum(h_ei)
        edge_support = torch.mm(edge_adj - I_edge, h)
        # edge_output: GRU(h_n(t-1), sum(h_ei))
        node_output = self.node_gru(edge_support, h)

        return torch.mm(I_edge, edge_output) + torch.mm(I_node, node_output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.nhidden) + ' -> ' \
               + str(self.nhidden) + ')'
