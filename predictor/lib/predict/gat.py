import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, in_channels, nhid, num_class, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.hidden_size = nhid

        self.fc_em_img = nn.Linear(in_channels, nhid)
        self.fc_em_obj = nn.Linear(in_channels, nhid)
        self.attentions = [GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid, nhid, dropout=0, alpha=alpha, concat=True)
        self.fusion_att = GraphAttentionLayer(nhid, nhid, dropout=0, alpha = alpha, concat=True)

        self.s_conv = nn.Conv1d(nhid,nhid,3,padding=1)
        self.t_conv = nn.Conv1d(nhid,1,3,padding=1)
        self.fc_fusion = nn.Linear(nhid*2, nhid)
        self.lstm_cell = nn.LSTMCell(nhid, nhid)
        self.fc_out = nn.Linear(nhid*2, num_class)
        nn.init.normal(self.fc_em_img.weight, mean=0, std=0.01)
        nn.init.normal(self.fc_em_obj.weight, mean=0, std=0.01)
        nn.init.normal(self.fc_fusion.weight, mean=0, std=0.01)
        nn.init.normal(self.fc_out.weight, mean=0, std=0.01)
        # for m in self.modules():
        #     if type(m) in [nn.GRU, nn.LSTMCell, nn.LSTM, nn.RNN]:
        #         k = 1.0/m.hidden_size
        #         for name, param in m.named_parameters():
        #             nn.init.normal(param,-k,k)
        self.t_conv.weight.data.normal_(mean=0.0, std=0.01)
        self.t_conv.bias.data.fill_(0)
        self.leaky_relu = nn.LeakyReLU(alpha)
                        

    def extract_feature(self, x, adj):
        N, V, C = x.size()
        x = x.view(V, C)
        x = self.fc_em_obj(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x,attn = self.out_att(x, adj)
        # x = x.sum(dim=0)
        # x = F.elu(x)
        # x = x.view(1,-1).contiguous()
        return x, attn
    
    def fusion_feature(self, x, ffeat_in, adj):
        ffeat_em = self.fc_em_img(ffeat_in)
        feat_all = torch.cat((ffeat_em, x), dim=0)
        feat_fusion, attn = self.fusion_att(feat_all, adj)
        feat_fusion = feat_fusion[0]
        feat_fusion = feat_fusion.view(1,-1).contiguous()
        # feat_fusion = torch.cat((ffeat_em,feat_fusion),dim=1)
        # feat_fusion = self.fc_fusion(feat_fusion)
        # feat_fusion = feat_fusion * h_prev + feat_fusion
        return feat_fusion, attn
    
    def classifer(self, x, h_prev):
        if len(h_prev)==0:
            h_prev = x
        else:
            h_prev = torch.cat((h_prev,x), dim=0)
        f = h_prev[:,:].permute(1,0)
        t = f.size(1)
        h = f.size(0)
        f = f.view(1,h,t)
        # f = self.s_conv(f)
        a = self.leaky_relu(self.t_conv(F.tanh(f)))
        a = a.view(1, t)
        a = F.softmax(a, dim=1)
        # a = F.sigmoid(a)
        f = torch.matmul(a, h_prev[:,:])
        ff = torch.cat((f,x),dim=1)
        res = self.fc_out(ff)
        return res, h_prev


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W1 = nn.Linear(in_features, out_features)
        self.W2 = nn.Linear(in_features, out_features)
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.normal_(self.W1.weight.data, mean=0, std=0.01)
        nn.init.normal_(self.W2.weight.data, mean=0, std=0.01)
        nn.init.constant_(self.W1.bias.data,0)
        nn.init.constant_(self.W2.bias.data,0)
        self.a = nn.Linear(out_features*2, 1)
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.normal_(self.a.weight.data, mean=0, std=0.01)
        nn.init.constant_(self.a.bias.data, 0)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h1 = self.W1(input)
        h2 = self.W2(input)
        N = h1.size()[0]

        a_input = torch.cat([h1.repeat(1, N).view(N * N, -1), h2.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leaky_relu(self.a(F.tanh(a_input))).squeeze(2)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.sigmoid(attention)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, input)

        return h_prime, attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == "__main__":
    model = GAT(in_channels=256,nhid=16,nclass=2,dropout=0.6,alpha=0.2,nheads=4)
    feat = torch.randn(10,256)
    adj = torch.ones(10,10)
    out = model(feat, adj)
    print(out)