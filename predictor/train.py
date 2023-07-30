from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import _init_paths
import time
import torch
import numpy as np
from dataset.data_feeder import DataFeeder
from utils.config import cfg_from_yaml
from utils.create_edge import create_1toN_adj, create_fc_adj
# from model.st_gcn import ST_GCN_18
# from model.gcn import GCN
from predict.gat import GAT
import math
import random
import os
from test import test

# def weights_init(model):
#     classname = model.__class__.__name__
#     if classname.find('Conv1d') != -1:
#         model.weight.data.normal_(0.0, 0.02)
#         if model.bias is not None:
#             model.bias.data.fill_(0)
#     elif classname.find('Conv2d') != -1:
#         model.weight.data.normal_(0.0, 0.02)
#         if model.bias is not None:
#             model.bias.data.fill_(0)
#     elif classname.find('BatchNorm') != -1:
#         model.weight.data.normal_(1.0, 0.02)
#         model.bias.data.fill_(0)

def train_epoch(e, model, data_loader, loss_fn, optimizer):
    train_tic = time.time()
    epoch_loss = 0.0
    fps = 20.0
    for name, num_det, det, feat, ffeat, label in data_loader:
        print(name[0])
        batch_tic = time.time()
        feat = feat.cuda()
        ffeat = ffeat.cuda()
        label = label.cuda()
        # N T V C
        batch_sz = feat.size()[0]
        
        T = feat.size()[1]

        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0).cuda()
        h_prev = []
        valid = False
        for i, n in enumerate(num_det[0,:]):
            if n <= 0:
                # print('no,obj')
                feat_in= torch.zeros(1,1024,1,1).cuda()
                continue
            # feat: N T V C
            valid = True
            feat_in = feat[:,i,:n,:]
            ffeat_in = ffeat[:,i,:]
            N = feat_in.size()[1]
            adj_obj = create_fc_adj(N).cuda()
            x,_  = model.extract_feature(feat_in, adj_obj)
            adj_att = create_1toN_adj(N + 1).cuda()
            x,_ = model.fusion_feature(x, ffeat_in, adj_att)
            output, h_prev = model.classifer(x, h_prev)
            # print(output)
            # res = torch.cat(all_pred, 0)
            pos_loss = -math.exp(-(T-i-1)/fps)*(-1 * loss_fn(output,label))
            neg_loss = loss_fn(output,label)
            loss = pos_loss*label.float() + neg_loss*(1-label).float()
            # loss = -math.exp(-(T-i-1)/fps)*(-1 * loss_fn(output,label))
            # loss = loss_fn(output, label)
            batch_loss = batch_loss + loss
        # print(output)
        # print('done')
        if valid:
            batch_loss.backward()
        optimizer.step()
        
        epoch_loss = epoch_loss + batch_loss.cpu().data
        batch_toc = time.time()
        # print('\tbatch loss:',batch_loss.cpu().data,',time:',batch_toc-batch_tic)
    train_toc = time.time()
    print('epoch loss:',epoch_loss,',time:',train_toc - train_tic)
    return

def save_checkpoint(e, model, optimizer, path):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_fname = path+"model-epoch-{}.pkl".format(e+1)
    torch.save({
            'epoch':e,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
    }, checkpoint_fname)
    return checkpoint_fname

def train(model_cfg, dataset_cfg, optimizer_cfg, 
        batch_size, total_epochs, training_hooks,
        gpus=1, workers = 0,
        resume_from=None, load_from=None, checkpoint_path=None):
    dataset = DataFeeder(**dataset_cfg)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                    shuffle=True, num_workers=workers)
    
    model = GAT(**model_cfg)
    # model.apply(weights_init)
    print(model)
    model = model.cuda()
    try:
        # set redution='none' to cal pos and neg loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    except TypeError:
        loss_fn = torch.nn.CrossEntropyLoss(reduce=False)

    optimizer = torch.optim.Adam(params=model.parameters(), **optimizer_cfg)
    # step = training_hooks['lr_step']
    # if isinstance(step, list):
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, step)
    # elif isinstance(step, int):
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step)
    # else:
    #     raise ValueError('not supported step for scheduler')
    if resume_from is not None:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
    else:
        start_epoch = 0

    for e in range(start_epoch, total_epochs):
        print('epoch',e,'start')
        train_epoch(e, model, data_loader, loss_fn, optimizer)
        # scheduler.step()
        if (e+1) % training_hooks['checkpoint_interval'] == 0:
            print('saving models')
            checkpoint_fname = save_checkpoint(e, model, optimizer, checkpoint_path)
        # config = cfg_from_yaml('cfgs/test.yaml')
        # config['checkpoint_path']=checkpoint_fname
        # test(**config)

if __name__ == "__main__":
    config = cfg_from_yaml('cfgs/train.yaml')
    train(**config)