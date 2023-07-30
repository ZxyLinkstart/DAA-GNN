from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import _init_paths
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from dataset.data_feeder import DataFeeder
from utils.config import cfg_from_yaml
from utils.create_edge import create_1toN_adj, create_fc_adj
from predict.gat import GAT
import math
import random
from eval import evaluation

def load_checkpoint(model, path):
    print("loading model from",path)
    checkpoint = torch.load(path)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

def test(model_cfg, dataset_cfg, 
        batch_size, gpus=1, workers = 0,
        resume_from=None, load_from=None, checkpoint_path=None):
    dataset = DataFeeder(**dataset_cfg)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                    shuffle=False, num_workers=workers)
    
    model = GAT(**model_cfg)
    model = model.cuda()
    load_checkpoint(model, checkpoint_path)
    # print(model)
    # model = model.cuda()
    model.eval()
    try:
        # set redution='none' to cal pos and neg loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    except TypeError:
        loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
    all_pred = []
    all_label = []
    test_loss = 0.0
    fps = 20.0
    for name, num_det, det, feat, ffeat, label in data_loader:
        # print(name[0])
        batch_tic = time.time()
        feat = feat.cuda()
        ffeat = ffeat.cuda()
        all_label.append(label.float().numpy())
        label = label.cuda()
        # N, T, V, C 
        batch_sz = feat.size()[0]
        T = feat.size()[1]
        h_prev = []
        batch_pred = []
        batch_loss = 0.0
        with torch.no_grad():
            for i, n in enumerate(num_det[0,:]):
                if n <= 0:
                    # print('no,obj')
                    output = np.zeros(1)
                    batch_pred.append(output)
                else:
                    # feat: N T V C
                    feat_in = feat[:,i,:n,:]
                    ffeat_in = ffeat[:,i,:]
                    N = feat_in.size()[1]
                    adj = create_fc_adj(N).cuda()
                    x,_ = model.extract_feature(feat_in, adj)
                    adj_att = create_1toN_adj(N + 1).cuda()
                    x,_ = model.fusion_feature(x, ffeat_in, adj_att)
                    output, h_prev = model.classifer(x, h_prev)
                    
                    pos_loss = -math.exp(-(T-i-1)/fps)*(-1 * loss_fn(output,label))
                    neg_loss = loss_fn(output,label)
                    loss = pos_loss*label.float() + neg_loss*(1-label).float()
                    batch_loss = batch_loss + loss

                    output = F.softmax(output,dim=-1)
                    output = output[:,1].cpu().numpy()
                    batch_pred.append(output)
                # print(output)
            batch_pred = np.stack(batch_pred,axis = 1)
            all_pred.append(batch_pred)
            test_loss = test_loss + batch_loss
        batch_toc = time.time()
        # print('time:',batch_toc - batch_tic)
    all_label = np.hstack(all_label)
    all_pred = np.vstack(all_pred)
    all_label = np.expand_dims(all_label,-1)
    all_pred = all_pred[:,:90]
    print(all_label.shape)
    print(all_pred.shape)
    print('loss', test_loss)
    evaluation(all_pred,all_label)

if __name__ == "__main__":
    config = cfg_from_yaml('cfgs/test.yaml')
    total_epochs = config['total_epochs']
    config.pop('total_epochs')
    checkpoint_path = config['checkpoint_path']
    for i in range(total_epochs):
        time_tic = time.time()
        config['checkpoint_path'] = checkpoint_path.format(i+1)
        while(not os.path.exists(config['checkpoint_path'])):
            print("waiting")
            time.sleep(60)
        test(**config)
        time_toc = time.time()
        print("time: ", time_toc-time_tic)