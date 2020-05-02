import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module): # adopted from https://github.com/Diego999/pyGAT
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha=0.2, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, node_adj, edge_adj):
        h = torch.mm(h, self.W) # (N, F)
        N = h.size()[0]

        h_plus = torch.mm((node_adj > 0).float(), h)
        h_minus = torch.mm((node_adj < 0).float(), h)
        a_input_plus = torch.cat((h_plus, h_minus), dim=1) # (N, 2F)
        a_input_minus = torch.cat((h_minus, h_plus), dim=1) # (N, 2F)
        e_plus = self.leakyrelu(torch.matmul(a_input_plus, self.a)).repeat(1, N) # (N, 1)
        e_minus = self.leakyrelu(torch.matmul(a_input_minus, self.a)).repeat(1, N) # (N, 1)

        zero_vec = -9e15*torch.ones_like(e_plus) # (N, N)
        attention = torch.where(edge_adj > 0, e_plus, zero_vec) # (N, N)
        attention = torch.where(edge_adj < 0, e_minus, attention) # (N, N)
        attention = F.softmax(attention, dim=1) # (N, N)
        h_prime = torch.matmul(attention * edge_adj, h) # (N, F)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class FactorGraphGRU(nn.Module):
    """
    Similar to GCN, except different GRU cells for nodes and edges (i.e. variables and factors)
    """
    def __init__(self, nhidden, msg_type, bias=True):
        super(FactorGraphGRU, self).__init__()
        self.nhidden = nhidden
        self.msg_type = msg_type
        self.bias = bias
        if self.msg_type == 'concat':
            self.edge_gru = nn.GRUCell(2*nhidden, nhidden, bias=self.bias)
        elif self.msg_type == 'diff':
            self.edge_gru = nn.GRUCell(nhidden, nhidden, bias=self.bias)
        else:
            assert False, 'Incorrect message type for model!'
        self.gat = GraphAttentionLayer(nhidden, nhidden)
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

        if self.msg_type == 'concat': # node_support: concat(h_n1, h_n2)
            node_support = torch.cat((torch.mm(((node_adj - I_node) > 0).float(), h), 
                torch.mm(((node_adj - I_node) < 0).float(), h)), dim=1)
        else: # node_support: h_n1 - h_n2
            node_support = torch.mm(node_adj - I_node, h)
        # edge_output: GRU(h_e(t-1), node_support)
        edge_output = self.edge_gru(node_support, h)

        # edge_support: sum(alpha_ei*h_ei)
        edge_support = self.gat(h, node_adj - I_node, edge_adj - I_edge)
        # node_output: GRU(h_n(t-1), edge_support)
        node_output = self.node_gru(edge_support, h)

        return torch.mm(I_edge, edge_output) + torch.mm(I_node, node_output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.nhidden) + ' -> ' \
               + str(self.nhidden) + ')'
