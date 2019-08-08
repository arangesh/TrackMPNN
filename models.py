import torch.nn as nn
import torch.nn.functional as F
from pointnet.model import *
from pygcn.layers import FactorGraphConvolution


class TrackMPNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(TrackMPNN, self).__init__()
        self.gc1 = FactorGraphConvolution(nfeat, nhid, bias=True, msg_type='concat')
        self.gc2 = FactorGraphConvolution(nhid, nhid, bias=True, msg_type='concat')
        self.gc3 = FactorGraphConvolution(nhid, nclass, bias=True, msg_type='concat')
        self.dropout = dropout
        self.pointnet = PointNetfeatsmall()

    def forward(self, x, node_adj, edge_adj):
        conv_hull =  x[:, -10:] # (N, 10)
        conv_hull = torch.cat((conv_hull.view(-1, 2, 5), torch.zeros(conv_hull.size()[0], 1, 5).cuda()), dim=1) # (N, 3, 5)
        # batchnorm does not work if N=1 (workaround)
        if conv_hull.size()[0] == 1:
            conv_hull_feat, _, _ = self.pointnet(torch.cat((conv_hull, conv_hull), 0)) # (2, 64)
            x = torch.cat((x[:, :-10], conv_hull_feat[:1, :]), dim=1) # (1, F-10+64)
        else:
            conv_hull_feat, _, _ = self.pointnet(conv_hull) # (N, 64)
            x = torch.cat((x[:, :-10], conv_hull_feat), dim=1) # (N, F-10+64)
        x = self.gc1(x, node_adj, edge_adj) # (N, 64)
        x = self.gc2(x, node_adj, edge_adj) # (N, 64)
        x = F.dropout(x, self.dropout, training=self.training) # (N, 64)
        x = self.gc3(x, node_adj, edge_adj) # (N, 2)

        return F.log_softmax(x, dim=1)
