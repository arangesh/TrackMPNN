import torch
import torch.nn as nn
import torch.sparse as sp

from models.layers import FactorGraphGRU


class TrackMPNN(nn.Module):
    def __init__(self, features, ncategories, nhidden, nattheads, msg_type):
        super(TrackMPNN, self).__init__()
        self.input_transforms = nn.ModuleList([])
        self.factor_grus = nn.ModuleList([])
        self.feature_idx = []
        self.nhidden = nhidden
        nfeatures = 0
        # for 2D+category features
        if '2d' in features:
            self.input_transforms.append(self.get_input_transform(ncategories+5, nhidden))
            self.factor_grus.append(FactorGraphGRU(nhidden, nattheads, msg_type, True))
            self.feature_idx.append(list(range(nfeatures, nfeatures+ncategories+5)))
            nfeatures += ncategories+5
        # for temporal features
        if 'temp' in features:
            self.input_transforms.append(self.get_input_transform(2, nhidden))
            self.factor_grus.append(FactorGraphGRU(nhidden, nattheads, msg_type, True))
            self.feature_idx.append(list(range(nfeatures, nfeatures+2)))
            nfeatures += 2
        # for visual features
        if 'vis' in features:
            self.input_transforms.append(self.get_input_transform(128, nhidden))
            self.factor_grus.append(FactorGraphGRU(nhidden, nattheads, msg_type, True))
            self.feature_idx.append(list(range(nfeatures, nfeatures+128)))
            nfeatures += 128

        self.output_transform_node = nn.Linear(len(self.feature_idx)*nhidden, 1, bias=True)
        self.output_transform_node.weight.data.normal_(mean=0.0, std=0.01)
        self.output_transform_node.bias.data.uniform_(+4.595, +4.595)

        self.output_transform_edge = nn.Linear(len(self.feature_idx)*nhidden, 1, bias=True)
        self.output_transform_edge.weight.data.normal_(mean=0.0, std=0.01)
        self.output_transform_edge.bias.data.uniform_(-4.595, -4.595)

        self.output_activation = nn.Sigmoid()

    def get_input_transform(self, n_in, n_out):
        input_transform_1 = nn.Linear(n_in, n_out, bias=True)
        input_transform_1.weight.data.normal_(mean=0.0, std=0.01)
        input_transform_1.bias.data.uniform_(0, 0)
        input_transform_2 = nn.Linear(n_out, n_out, bias=True)
        input_transform_2.weight.data.normal_(mean=0.0, std=0.01)
        input_transform_2.bias.data.uniform_(0, 0)
        return nn.Sequential(input_transform_1, nn.BatchNorm1d(n_out), nn.ReLU(), input_transform_2)

    def forward(self, x, h_in, node_adj, edge_adj):
        I_node = torch.diag(torch.diag(node_adj.to_dense())).to_sparse() # (N', N')
        I_edge = torch.diag(torch.diag(edge_adj.to_dense())).to_sparse() # (N', N')

        if x.size()[0] > 0:
            xs = [self.input_transforms[_](x[:, idx]) for _, idx in enumerate(self.feature_idx)] # [(N'-N, nhidden),]

            hs_update = [sp.mm(I_node.to_dense()[-_x.size()[0]:, -_x.size()[0]:].to_sparse(), _x) for _x in xs] # [(N'-N, nhidden),]
            if h_in is None:
                hs = hs_update # [(N'-N, nhidden),]
            else:
                hs_in = torch.split(h_in, self.nhidden, dim=1) # [(N, nhidden),]
                hs = [torch.cat((_h[0], _h[1]), dim=0) for _h in zip(hs_in, hs_update)] # [(N', nhidden),]
        else:
            hs = torch.split(h_in, self.nhidden, dim=1) # [(N, nhidden),]

        hs_att_out = [self.factor_grus[_](_h, node_adj, edge_adj) for _, _h in enumerate(hs)] # [(N', nhidden),]
        hs_out, attention = zip(*hs_att_out)
        h_out = torch.cat(hs_out, dim=1) # (N', 3*nhidden)
        y = sp.mm(I_node, self.output_transform_node(h_out)) + sp.mm(I_edge, self.output_transform_edge(h_out)) # (N', 1)

        return self.output_activation(y), y, h_out, attention
