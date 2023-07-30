import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gconv import ConvTemporalGraphical

class GCN(nn.Module):
    r"""graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 atten_hidden,
                 data_bn=True,
                 **kwargs):
        super(GCN, self).__init__()

        # load graph
        # self.graph = Graph(**graph_cfg)
        # A = torch.tensor(
        #     self.graph.A, dtype=torch.float32, requires_grad=False)
        # self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = 2
        temporal_kernel_size = 9
        # self.data_bn = nn.BatchNorm1d(in_channels) if data_bn else lambda x: x
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.gcn_networks = nn.ModuleList((
            gcn_block(in_channels, 512, atten_hidden, spatial_kernel_size),
            gcn_block(512, 512, atten_hidden, spatial_kernel_size),
            gcn_block(512, 256, atten_hidden, spatial_kernel_size),
            gcn_block(256, 256, atten_hidden, spatial_kernel_size, activate=False),
        ))
        self.tcn_networks = nn.ModuleList((
            tcn_block(256, 256, temporal_kernel_size, 2, **kwargs0),
            tcn_block(256, 256, temporal_kernel_size, 1, **kwargs),
            tcn_block(256, 256, temporal_kernel_size, 2, **kwargs),
            tcn_block(256, 256, temporal_kernel_size, 1, **kwargs),
        ))
        
        self.edge_importance = [1] * len(self.gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

        self.graph = Graph(**graph_cfg)

    def extract_feature(self, x, det):
        A = self.graph.cal_A(det).cuda()
        # data normalization
        N, C, V, M = x.size()
        # N, C, T, V, M = x.size()
        # x = x.view(N, C, 1, V, M)
        # x = x.permute(0, 4, 1, 3, 2).contiguous()
        # x = x.view(N * M, C, V)
        # x = self.data_bn(x)
        # x = x.view(N, M, C, V, 1)
        # x = x.permute(0, 1, 2, 4, 3).contiguous()
        # x = x.view(N * M, C, 1, V)

        x = x.permute(0,3,1,2).contiguous()
        x = x.view(N*M, C, 1, V)

        # forwad
        for gcn,importance in zip(self.gcn_networks,self.edge_importance):
            x, _ = gcn(x, A*importance)
        
        # x = x.view(N*M, -1, V)
        x = x.sum(dim=3)
        x = x.view(N*M, -1, 1, 1).contiguous()
        return x
    
    def forward(self, x):
        N, C, T, V = x.size()
        for tcn in self.tcn_networks:
            x = tcn(x)
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        # x = x.view(N, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

import numpy as np
def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

class Graph(object):
    r"""Using bbox of objects to calculate adj between graph nodes.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the linear layer
        alpha (int, optional): Alpha of the leaky relu. Default: 0.2
        dropout (int, optional): Dropout rate of the final output. Default: 0
        data_bn (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input det sequence in :math:`(1, V, T_{in}, in_channels//T)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(V, V)` format

        where
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """
    def __init__(self, mode='freq', freq_thres = 0.05):
        super(Graph, self).__init__()
        # class_names = ['person','bike','motor','car','bus']
        obj_freq = [[33366,  1976, 31047, 29779,  4043],
                        [ 1976, 2857,  2279,  2515,   111],
                        [31047,  2279, 44078, 39657,  4501],
                        [29779,  2515, 39657, 56073,  5974],
                        [ 4043,   111,  4501,  5974,  6365]]
        multi_inst = [16398,   361, 25203, 43608,   552.]
        self.obj_freq = np.array(obj_freq)
        self.multi_inst = np.array(multi_inst)
        self.total_inst = self.obj_freq.diagonal().sum()
        self.threshold = freq_thres
        self.max_width = 1280
        self.max_height = 720

    def cal_A(self, det):
        det = det[0]
        num_nodes = det.size()[0]
        self_A = np.eye(num_nodes)
        freq_A = np.zeros((num_nodes, num_nodes))
        dist_A = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            cls_id_i = int(det[i,-1].item())
            box_i = det[i,:4].cpu().numpy()
            for j in range(num_nodes):
                if j==i:
                    continue
                cls_id_j = int(det[j,-1].item())
                box_j = det[j,:4].cpu().numpy()
                if cls_id_i == cls_id_j:
                    freq_A[i,j] = self.multi_inst[cls_id_i]*1.0/self.obj_freq[cls_id_i,cls_id_i]
                else:
                    freq_A[i,j] = self.cal_freq(cls_id_i, cls_id_j)
                dist_A[i,j] = self.cal_dist(box_i, box_j)
        dist_A = normalize_digraph(dist_A)
        freq_A = normalize_digraph(freq_A + self_A)
        A = np.stack((freq_A, dist_A))
        return torch.tensor(A, dtype=torch.float32, requires_grad=False)
    
    def cal_freq(self, cls_id_a, cls_id_b):
        if cls_id_a == cls_id_b:
            return self.multi_inst[cls_id_a]*1.0/self.obj_freq[cls_id_a,cls_id_a]
        else:
            return self.obj_freq[cls_id_a,cls_id_b]*1.0/self.obj_freq[cls_id_b,cls_id_b]
    def cal_dist(self, box_a, box_b):
        x1a,y1a,x2a,y2a = box_a
        x1b,y1b,x2b,y2b = box_b
        dis_x = (x1a+x2a-x1b-x2b)*1.0/(self.max_width/2)
        dis_x = dis_x if dis_x>0 else -dis_x
        dis_y = (y1a+y2a-y1b-y2b)*1.0/(self.max_height/2)
        dis_y = dis_y if dis_y>0 else -dis_y
        res = dis_y*dis_y + dis_x*dis_x
        res = np.sqrt(res)
        # res = dis_y+ dis_x
        res = np.exp(-res)
        return res

class gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 atten_hidden,
                 kernel_size,
                 activate=True):
        super(gcn_block, self).__init__()

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size)
        self.attention_block = attention_block(in_channels, atten_hidden)
        if activate:
            self.activation = nn.ReLU()
        else:
            self.activation = lambda x: x

    def forward(self, x, A):
        # attention = self.attention_block(x)
        x, A = self.gcn(x, A)
        return self.activation(x), A

class attention_block(nn.Module):
    r"""Using features of objects to calculate attention of graph nodes.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the linear layer
        alpha (int, optional): Alpha of the leaky relu. Default: 0.2
        dropout (int, optional): Dropout rate of the final output. Default: 0
        data_bn (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input det sequence in :math:`(1, V, T_{in}, in_channels//T)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(V, V)` format

        where
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """
    def __init__(self, in_channels, out_channels, alpha = 0.2, dropout=0):
        super(attention_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(in_channels, self.out_channels)))
        nn.init.normal_(self.W.data, mean=0, std=0.01)
        self.a = nn.Parameter(torch.zeros(size=(2*self.out_channels, 1)))
        nn.init.normal_(self.a.data, mean=0, std=0.01)

        # self.det_bn = nn.BatchNorm1d(in_channels) if data_bn else lambda x: x
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x):
        # N, V, T, C = x.size()
        N, C, T, V = x.size()
        x = x.permute(0,2,3,1).contiguous()
        x = x.view(T*V,C)
        h = torch.mm(x, self.W)
        V = h.size()[0]
        a_input = torch.cat([h.repeat(1, V).view(V * V, -1), h.repeat(V, 1)], dim=1).view(V, -1, 2 * self.out_channels)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(e > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        return attention

class tcn_block(nn.Module):
    r"""Applies a temporal convolution over an input sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the temporal convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=False):
        super(tcn_block, self).__init__()

        assert kernel_size % 2 == 1
        padding = ((kernel_size - 1) // 2, 0)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_size, 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        res = self.residual(x)
        x = self.tcn(x) + res

        return self.relu(x)