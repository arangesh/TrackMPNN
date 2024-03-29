import torch
import torch.nn as nn
import torch.sparse as sp
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module): # adapted from https://github.com/Diego999/pyGAT
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha=0.2, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W_att = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_att.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, h, node_adj, edge_adj):
        h_att = torch.mm(h, self.W_att) # (N, F)
        N = h.size()[0]

        h_plus = sp.mm((node_adj > 0).float().to_sparse(), h_att)
        h_minus = sp.mm((node_adj < 0).float().to_sparse(), h_att)
        a_input = torch.abs(h_plus - h_minus) # (N, F)
        e = self.leakyrelu(torch.matmul(a_input, self.a)).transpose(0, 1).repeat(N, 1) # (N, N)

        attention = torch.where(edge_adj != 0, e, torch.tensor(-9e15).to(e.device)) # (N, N)
        attention = F.softmax(attention, dim=1) # (N, N)
        attention = self.dropout(attention)
        h_prime = sp.mm((attention * edge_adj).to_sparse(), h) # (N, F)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class FactorGraphGRU(nn.Module):
    """
    Similar to GCN, except different GRU cells for nodes and edges (i.e. variables and factors)
    """
    def __init__(self, nhidden, nattheads=0, msg_type='diff', bias=True):
        super(FactorGraphGRU, self).__init__()
        self.nhidden = nhidden
        self.msg_type = msg_type
        self.nattheads = nattheads
        self.bias = bias
        if self.msg_type == 'concat':
            self.edge_gru = nn.GRUCell(2*nhidden, nhidden, bias=self.bias)
        elif self.msg_type == 'diff':
            self.edge_gru = nn.GRUCell(nhidden, nhidden, bias=self.bias)
        else:
            assert False, 'Incorrect message type for model!'
        if self.nattheads <= 0:
            self.gat = None
        else:
            self.gat = nn.ModuleList([GraphAttentionLayer(nhidden, nhidden) for _ in range(self.nattheads)])
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
        I_node = torch.diag(torch.diag(node_adj.to_dense())).to_sparse() # (N', N')
        I_edge = torch.diag(torch.diag(edge_adj.to_dense())).to_sparse() # (N', N')
        node_adj_norm = node_adj - I_node
        edge_adj_norm = edge_adj - I_edge

        if self.msg_type == 'concat': # node_support: concat(h_n1, h_n2)
            node_adj_pos = (node_adj_norm.to_dense() > 0).float().to_sparse()
            node_adj_neg = (node_adj_norm.to_dense() < 0).float().to_sparse()
            node_support = torch.cat((sp.mm(node_adj_pos, h), sp.mm(node_adj_neg, h)), dim=1)
        else: # node_support: h_n1 - h_n2
            node_support = sp.mm(node_adj_norm, h)
        # edge_output: GRU(h_e(t-1), node_support)
        edge_output = self.edge_gru(node_support, h)

        # edge_support: sum(alpha_ei*h_ei)
        node_adj_norm, edge_adj_norm = node_adj_norm.to_dense(), edge_adj_norm.to_dense()
        if self.gat is None:
            attention = None
            edge_support = sp.mm(edge_adj_norm.to_sparse(), h) # (N, F)
        else:
            attention = []
            edge_support, _att = self.gat[0](h, node_adj_norm, edge_adj_norm)
            attention.append(_att)
            for i in range(1, self.nattheads):
                _e_supp, _att = self.gat[i](h, node_adj_norm, edge_adj_norm)
                edge_support += _e_supp
                attention.append(_att)
            edge_support = edge_support / self.nattheads
        # node_output: GRU(h_n(t-1), edge_support)
        node_output = self.node_gru(edge_support, h)

        return sp.mm(I_edge, edge_output) + sp.mm(I_node, node_output), attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.nhidden) + ' -> ' \
               + str(self.nhidden) + ')'
