import torch
import torch.nn as nn

from models.pointnet.model import PointNetfeatsmall
from models.layers import FactorGraphGRU


class TrackMPNN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(TrackMPNN, self).__init__()
        self.pointnet = PointNetfeatsmall()

        self.input_transform_1 = nn.Linear(nfeat, nhid, bias=True)
        self.input_transform_1.weight.data.normal_(mean=0.0, std=0.01)
        self.input_transform_1.bias.data.uniform_(0, 0)
        self.input_transform_2 = nn.Linear(nhid, nhid, bias=True)
        self.input_transform_2.weight.data.normal_(mean=0.0, std=0.01)
        self.input_transform_2.bias.data.uniform_(0, 0)
        self.input_transform = nn.Sequential(self.input_transform_1, nn.ReLU(), self.input_transform_2)

        self.factor_gru1 = FactorGraphGRU(nhid, bias=True, msg_type='concat')

        self.output_transform_node = nn.Linear(nhid, 1, bias=True)
        self.output_transform_node.weight.data.normal_(mean=0.0, std=0.01)
        self.output_transform_node.bias.data.uniform_(+4.595, +4.595)

        self.output_transform_edge = nn.Linear(nhid, 1, bias=True)
        self.output_transform_edge.weight.data.normal_(mean=0.0, std=0.01)
        self.output_transform_edge.bias.data.uniform_(-4.595, -4.595)

        self.output_activation = nn.Sigmoid()

    def forward(self, x, h_in, node_adj, edge_adj):
        I_node = torch.diag(torch.diag(node_adj)) # (N', N')
        I_edge = torch.diag(torch.diag(edge_adj)) # (N', N')

        if x.size()[0] > 0: 
            conv_hull =  x[:, -10:] # (N'-N, 10)
            if next(self.parameters()).is_cuda:
                conv_hull = torch.cat((conv_hull.view(-1, 2, 5), torch.zeros(conv_hull.size()[0], 1, 5).cuda()), dim=1) # (N'-N, 3, 5)
            else:
                conv_hull = torch.cat((conv_hull.view(-1, 2, 5), torch.zeros(conv_hull.size()[0], 1, 5)), dim=1) # (N'-N, 3, 5)
            # batchnorm does not work if N=1 (workaround)
            if conv_hull.size()[0] == 1:
                conv_hull_feat, _, _ = self.pointnet(torch.cat((conv_hull, conv_hull), 0)) # (2, 64)
                x = torch.cat((x[:, :-10], conv_hull_feat[:1, :]), dim=1) # (1, F-10+64)
            else:
                conv_hull_feat, _, _ = self.pointnet(conv_hull) # (N'-N, 64)
                x = torch.cat((x[:, :-10], conv_hull_feat), dim=1) # (N'-N, F-10+64)

            h_update = torch.mm(I_node[-x.size()[0]:, -x.size()[0]:], self.input_transform(x)) # (N'-N, 64)
            if h_in is None: # (N, 64)
                h = h_update # (N', 64)
            else:
                h = torch.cat((h_in, h_update), dim=0) # (N', 64)
        else:
            h = h_in

        h_out = self.factor_gru1(h, node_adj, edge_adj) # (N', 64)
        y = torch.mm(I_node, self.output_transform_node(h_out)) + torch.mm(I_edge, self.output_transform_edge(h_out)) # (N', 1)

        return self.output_activation(y), h_out
