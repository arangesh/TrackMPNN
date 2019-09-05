import torch
import torch.nn as nn

from models.pointnet.model import PointNetfeatsmall
from models.layers import FactorGraphConvolution, FactorGraphResidual


class TrackMPNN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(TrackMPNN, self).__init__()
        self.pointnet = PointNetfeatsmall()

        # Options for the activation function incldue: nn.ReLU(), nn.LeakyReLU(), nn.PReLU(), nn.Sigmoid(), nn.Tanh()
        self.gc1 = FactorGraphConvolution(nfeat, nhid, bias=True, msg_type='concat', activation=nn.ReLU())
        self.gc2 = FactorGraphConvolution(nhid, nhid, bias=True, msg_type='concat', activation=nn.ReLU())
        self.gc3 = FactorGraphConvolution(nhid, 1, bias=True, msg_type='concat', activation=None)
        nn.init.constant_(self.gc3.node_bias.data, -4.595) # -log((1 - p)/ p) with p=0.01 from Focal Loss paper
        nn.init.constant_(self.gc3.edge_bias.data, +4.595) # +log((1 - p)/ p) with p=0.01 from Focal Loss paper

        #self.gc1 = FactorGraphConvolution(nfeat, nhid, bias=True, msg_type='concat', activation=nn.ReLU())
        #self.gc2 = FactorGraphResidual(nhid, bias=True, msg_type='concat', activation=nn.ReLU())
        #self.gc3 = FactorGraphResidual(nhid, bias=True, msg_type='concat', activation=nn.ReLU())
        #self.gc4 = FactorGraphConvolution(nhid, 1, bias=True, msg_type='concat', activation=nn.ReLU())
        #nn.init.constant_(self.gc4.node_bias.data, -4.595) # -log((1 - p)/ p) with p=0.01 from Focal Loss paper
        #nn.init.constant_(self.gc4.edge_bias.data, +4.595) # +log((1 - p)/ p) with p=0.01 from Focal Loss paper

        self.output_activation = nn.Sigmoid()

    def forward(self, x, node_adj, edge_adj):
        conv_hull =  x[:, -10:] # (N, 10)
        if next(self.parameters()).is_cuda:
            conv_hull = torch.cat((conv_hull.view(-1, 2, 5), torch.zeros(conv_hull.size()[0], 1, 5).cuda()), dim=1) # (N, 3, 5)
        else:
            conv_hull = torch.cat((conv_hull.view(-1, 2, 5), torch.zeros(conv_hull.size()[0], 1, 5)), dim=1) # (N, 3, 5)
        # batchnorm does not work if N=1 (workaround)
        if conv_hull.size()[0] == 1:
            conv_hull_feat, _, _ = self.pointnet(torch.cat((conv_hull, conv_hull), 0)) # (2, 64)
            x = torch.cat((x[:, :-10], conv_hull_feat[:1, :]), dim=1) # (1, F-10+64)
        else:
            conv_hull_feat, _, _ = self.pointnet(conv_hull) # (N, 64)
            x = torch.cat((x[:, :-10], conv_hull_feat), dim=1) # (N, F-10+64)

        x = self.gc1(x, node_adj, edge_adj) # (N, 64)
        x = self.gc2(x, node_adj, edge_adj) # (N, 64)
        x = self.gc3(x, node_adj, edge_adj) # (N, 1)

        #diag_ind = (torch.arange(node_adj.size()[0]), torch.arange(node_adj.size()[0]))
        #node_adj_res = node_adj
        #edge_adj_res = edge_adj
        #node_adj_res[diag_ind] = 0
        #edge_adj_res[diag_ind] = 0
        #x = self.gc1(x, node_adj, edge_adj) # (N, 64)
        #x = self.gc2(x, node_adj_res, edge_adj_res) # (N, 64)
        #x = self.gc3(x, node_adj_res, edge_adj_res) # (N, 64)
        #x = self.gc4(x, node_adj, edge_adj) # (N, 1)

        return self.output_activation(x)
