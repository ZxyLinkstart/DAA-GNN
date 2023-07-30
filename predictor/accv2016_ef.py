from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import _init_paths
import cv2
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from dataset.data_feeder import DataFeeder

############### Global Parameters ###############
# path
train_path = './data/ef_detres_0113_20p/training/'
test_path = './data/ef_detres_0113_20p/testing/'
demo_path = './data/ef_detres_0113_20p/testing/'
default_model_path = './demo_model/demo_model'
save_path = './models/predictor_accv16_ef/'
video_path = './data/videos/testing/positive/'
# batch_number
train_num = 128
test_num = 46


############## Train Parameters #################

# Parameters
learning_rate = 0.0001
n_epochs = 40
batch_size = 1
display_step = 10

# Network Parameters
n_input = 1024 # fc6 or fc7(1*4096)
n_detection = 20 # number of object of each image (include image features)
n_hidden = 512 # hidden layer num of LSTM
n_img_hidden = 256 # embedding image features
n_att_hidden = 256 # embedding object features
n_classes = 2 # has accident or not
n_frames = 100 # number of frame in each video
##################################################

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='accident_LSTM')
    parser.add_argument('--mode',dest = 'mode',help='train or test',default = 'train')
    parser.add_argument('--model',dest = 'model',default= default_model_path)
    parser.add_argument('--resume',dest = 'resume',default= None)
    parser.add_argument('--gpu',dest = 'gpu',default= '0')
    args = parser.parse_args()

    return args

class Model(nn.Module):
    def __init__(self, keep=0.5, device='cpu'):
        super(Model,self).__init__()
        self.device = device
        self.keep = keep

        # define weights
        self.weights_em_obj = nn.Parameter(torch.zeros(n_input, n_att_hidden))
        self.weights_em_img = nn.Parameter(torch.zeros(n_input, n_img_hidden))
        self.weights_att_w  = nn.Parameter(torch.zeros(n_att_hidden,1))
        self.weights_att_wa = nn.Parameter(torch.zeros(n_hidden, n_att_hidden))
        self.weights_att_ua = nn.Parameter(torch.zeros(n_att_hidden, n_att_hidden))
        self.weights_out    = nn.Parameter(torch.zeros(n_hidden, n_classes))

        nn.init.normal_(self.weights_em_obj.data,mean=0.0,std=0.01)
        nn.init.normal_(self.weights_em_img.data,mean=0.0,std=0.01)
        nn.init.normal_(self.weights_att_w.data,mean=0.0,std=0.01)
        nn.init.normal_(self.weights_att_wa.data,mean=0.0,std=0.01)
        nn.init.normal_(self.weights_att_ua.data,mean=0.0,std=0.01)
        nn.init.normal_(self.weights_out.data,mean=0.0,std=0.01)

        self.biases_em_obj = nn.Parameter(torch.zeros(n_att_hidden))
        self.biases_em_img = nn.Parameter(torch.zeros(n_img_hidden))
        self.biases_att_ba = nn.Parameter(torch.zeros(n_att_hidden))
        self.biases_out    = nn.Parameter(torch.zeros(n_classes))

        # nn.init.normal_(self.biases_em_obj.data,mean=0.0,std=0.01)
        nn.init.constant_(self.biases_em_obj.data,0)
        # nn.init.normal_(self.biases_em_img.data,mean=0.0,std=0.01)
        nn.init.constant_(self.biases_em_img.data,0)
        nn.init.constant_(self.biases_att_ba.data,0)
        # nn.init.normal_(self.biases_out.data,mean=0.0,std=0.01)
        nn.init.constant_(self.biases_out.data,0)

        # define a lstm cell
        self.lstm_cell = nn.LSTMCell(n_img_hidden+n_att_hidden, n_hidden)
        # self.lstm_cell = nn.LSTM(n_img_hidden+n_att_hidden, n_hidden)

        # for m in self.modules():
        #     if type(m) in [nn.GRU, nn.LSTM, nn.RNN, nn.LSTMCell]:
        #         for name, param in m.named_parameters():
        #             torch.nn.init.normal_(param.data,mean=0,std=0.01)
        
        # torch >= 0.4.1
        # self.loss = nn.CrossEntropyLoss(reduction='none')
        try:
            # set redution='none' to cal pos and neg loss
            self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        except TypeError:
            self.loss = torch.nn.CrossEntropyLoss(reduce=False)

        self.to(device)

    def forward(self, X, y):
        # input features (Faster-RCBB fc7)
        N = X.size()[0]
        istate = (torch.zeros( N, self.lstm_cell.hidden_size).float().to(self.device),
                torch.zeros( N, self.lstm_cell.hidden_size).float().to(self.device))
        h_prev = torch.zeros( N, n_hidden).float().to(self.device)
        loss = 0.0

        zeros_object = X[:,:,1:n_detection,:].permute(1,2,0,3).sum(3)!=0
        zeros_object = zeros_object.float()
        T = X.size()[1]
        for i in range(T):
            x = X[:,i,:n_detection,:].permute(1,0,2) # permute n_steps and batch_size (n x b x h)
            # frame embedded
            image = torch.matmul(x[0], self.weights_em_img)+self.biases_em_img # b x h
            # object embedded
            n_object = x[1:n_detection,:,:].contiguous().view(-1,n_input) # (n_steps*batch_size, n_input)
            n_object = torch.matmul(n_object, self.weights_em_obj)+self.biases_em_obj # (n-1 x b) x h
            n_object = n_object.view(n_detection-1,  N, n_att_hidden) # n-1 x b x h
            n_object = n_object*(zeros_object[i].unsqueeze(2))

            # object attention
            brcst_w = self.weights_att_w.unsqueeze(0).repeat(n_detection-1,1,1) # n x h x 1

            image_part = torch.matmul(n_object, self.weights_att_ua.unsqueeze(0).repeat(n_detection-1,1,1)) + self.biases_att_ba # n x b x h
            e = torch.tanh(torch.matmul(h_prev,self.weights_att_wa) + image_part) # n x b x h
            # the probability of each object
            alphas = F.softmax(torch.matmul(e,brcst_w).sum(2),dim=0) * zeros_object[i]
            # weighting sum
            attention_list = (alphas.unsqueeze(2)) * n_object
            attention = attention_list.sum(0) # b x h
            # concat frame & object
            fusion = torch.cat((image,attention),dim=1)
        
            h_prev,c_prev = self.lstm_cell(fusion, istate)
            h_prev = F.dropout(h_prev, p= 1- self.keep, training = self.training)
            # save prev hidden state of LSTM
            istate = (h_prev,c_prev)
            outputs =  h_prev
            # FC to output
            pred = torch.matmul(outputs, self.weights_out)+self.biases_out # b x n_classes
            # save the predict of each time step
            if i == 0:
                soft_pred = F.softmax(pred, dim=-1).permute(1,0)[1].view( N,1)
                all_alphas = alphas.unsqueeze(0)
            else:
                temp_soft_pred = F.softmax(pred,dim=-1).permute(1,0)[1].view( N,1)
                soft_pred = torch.cat((soft_pred, temp_soft_pred),1)
                temp_alphas = alphas.unsqueeze(0)
                all_alphas = torch.cat((all_alphas, temp_alphas),0)

            yl = torch.argmax(y,dim=-1)
            # positive example (exp_loss)
            pos_loss = -math.exp(-(T-i-1)/30.0)* (-1 * self.loss(pred, y))
            # negative example
            neg_loss = self.loss(pred,y)

            temp_loss = (pos_loss*y.float() + neg_loss*(1-y).float()).mean()
            loss = loss + temp_loss

        return loss, soft_pred, all_alphas

def train(resume_path):
    # build model
    print("Building model...")
    device = 'cpu'
    if torch.cuda.is_available():
        device='cuda'
    model = Model(device = device)
    print(model)
    dataset = DataFeeder(data_path=train_path)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                    shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # mkdir folder for saving model
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    start_epoch = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
    # keep train unitl reach max iterations
    # start training
    print("Starting training ...")
    for epoch in range(start_epoch,n_epochs):
        print("Epoch:", epoch+1, "start...")
        tStart_epoch = time.time()
        model.train()
        # random chose batch.npz
        epoch_loss = []
        tStop_epoch = time.time()
        batch = 0
        for name, _, _, feat, ffeat, label in data_loader:
            # print(name[0])
            ffeat = ffeat.to(device)
            feat = feat.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            N, T, C = ffeat.size()
            ffeat = ffeat.view(N, T, 1, C)
            feat_all = torch.cat((ffeat,feat), dim = 2)
            batch_loss,_,_ = model(feat_all, label)
            print("\t training", batch, "th batch, loss:", batch_loss.item())
            batch_loss.backward()
            optimizer.step()
            batch_loss = batch_loss.data.cpu().numpy()
            epoch_loss.append(batch_loss/batch_size)
        # print one epoch
        print("Epoch:", epoch+1, " done. Loss:", np.mean(epoch_loss))
        tStop_epoch = time.time()
        print("Epoch Time Cost:", round(tStop_epoch-tStart_epoch, 2),"s")
        sys.stdout.flush()
        if (epoch+1)%1 == 0:
            torch.save({
                    'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':np.mean(epoch_loss)
                },save_path+"model-epoch-{}.pkl".format(epoch+1))
            # torch.save(model.state_dict(), save_path+"model-epoch-{}.pkl".format(epoch+1))
            # print("Training")
            # test_all(model, train_num, train_path,device)
            # if epoch<=4:
            #     continue
            print("Testing")
            test_all(model, test_path, device)
    print("Optimization Finished")
    torch.save({
                    'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':np.mean(epoch_loss)
                },save_path+"model-final.pkl")
    # torch.save(model.state_dict(), save_path+"model-final.pkl")

def test_all(model, path, device):
    model.eval()
    total_loss = 0.0
    dataset = DataFeeder(data_path=path)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                    shuffle=False, num_workers=0)
    num_batch = 0
    length = []
    all_pred = []
    all_label = []
    for name, _, _, feat, ffeat, label in data_loader:
        # print(name[0])
        ffeat = ffeat.to(device)
        feat = feat.to(device)
        all_label.append(label.float().numpy())
        label = label.to(device)
        N, T, C = ffeat.size()
        length.append(T)
        ffeat = ffeat.view(N, T, 1, C)
        feat_all = torch.cat((ffeat,feat), dim = 2)
        with torch.no_grad():
            temp_loss, pred, _ = model(feat_all, label)
            temp_loss = temp_loss.data.cpu().numpy()
            pred = pred.data.cpu().numpy()
        total_loss += temp_loss/N
        num_batch +=1
        all_pred.append(pred)

    all_label = np.hstack(all_label)
    all_label = np.expand_dims(all_label,-1)
    # all_pred = np.vstack(all_pred)
    total_time = max(length)
    all_pred_tmp = np.zeros((len(all_pred),total_time))
    for idx, vid in enumerate(length):
        all_pred_tmp[idx,total_time-vid:] = all_pred[idx][:]
    all_pred = np.array(all_pred_tmp)
    # all_pred = all_pred[:,:90]
    print(all_label.shape)
    print(all_pred.shape)
    print('loss', total_loss)
    evaluation(all_pred,all_label,total_time=total_time,length=length)

def evaluation(all_pred,all_labels, total_time = 90, vis = False, length = None):
    ### input: all_pred (N x total_time) , all_label (N,)
    ### where N = number of videos, fps = 20 , time of accident = total_time
    ### output: AP & Time to Accident

    if length is not None:
        all_pred_tmp = np.zeros(all_pred.shape)
        for idx, vid in enumerate(length):
                all_pred_tmp[idx,total_time-vid:] = all_pred[idx,total_time-vid:]
        all_pred = np.array(all_pred_tmp)
        temp_shape = sum(length)
    else:
        length = [total_time] * all_pred.shape[0]
        temp_shape = all_pred.shape[0]*total_time
    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0
    for Th in sorted(all_pred.flatten()):
        if length is not None and Th == 0:
                continue
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0
        for i in range(len(all_pred)):
            tp =  np.where(all_pred[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                time += tp[0][0] / float(length[i])
                counter = counter+1
            Tp_Fp += float(len(np.where(all_pred[i]>=Th)[0])>0)
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0:
            Recall[cnt] = np.nan
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            Time[cnt] = np.nan
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _,rep_index = np.unique(Recall,return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]

    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    print("Average Precision= " + "{:.4f}".format(AP) + " ,mean Time to accident= " +"{:.4}".format(np.mean(new_Time) * 5))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    print("Recall@80%, Time to accident= " +"{:.4}".format(sort_time[np.argmin(np.abs(sort_recall-0.8))] * 5))

    ### visualize

    if vis:
        plt.plot(new_Recall, new_Precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(AP))
        plt.show()
        plt.clf()
        plt.plot(new_Recall, new_Time, label='Recall-mean_time curve')
        plt.xlabel('Recall')
        plt.ylabel('time')
        plt.ylim([0.0, 5])
        plt.xlim([0.0, 1.0])
        plt.title('Recall-mean_time' )
        plt.show()


def vis(model_path):
    # build model
    device = 'cpu'
    if torch.cuda.is_available():
        device='cuda'
    model = Model(device = device)
    # restore model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # load data
    for num_batch in range(1,test_num):
        file_name = '{:03d}'.format(num_batch)
        all_data = np.load(demo_path+'batch_'+file_name+'.npz')
        data = torch.from_numpy(all_data['data']).float().to(device)
        labels = torch.from_numpy(all_data['labels']).float().to(device)
        det = all_data['det']
        ID = all_data['ID']
        # run result
        with torch.no_grad:
            all_loss, pred, weight = model(data, labels)
            all_loss = all_loss.data.cpu().numpy()
            pred = pred.data.cpu().numpy()
            weight = weight.data.cpu().numpy()
        file_list = sorted(os.listdir(video_path))
        for i in range(len(ID)):
            if labels[i][1] == 1 :
                plt.figure(figsize=(14,5))
                plt.plot(pred[i,0:90],linewidth=3.0)
                plt.ylim(0, 1)
                plt.ylabel('Probability')
                plt.xlabel('Frame')
                plt.show()
                file_name = ID[i]
                bboxes = det[i]
                new_weight = weight[:,:,i]*255
                counter = 0
                cap = cv2.VideoCapture(video_path+file_name+'.mp4')
                ret, frame = cap.read()
                while(ret):
                    attention_frame = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.uint8)
                    now_weight = new_weight[counter,:]
                    new_bboxes = bboxes[counter,:,:]
                    index = np.argsort(now_weight)
                    for num_box in index:
                        if now_weight[num_box]/255.0>0.4:
                            cv2.rectangle(frame,(new_bboxes[num_box,0],new_bboxes[num_box,1]),(new_bboxes[num_box,2],new_bboxes[num_box,3]),(0,255,0),3)
                        else:
                            cv2.rectangle(frame,(new_bboxes[num_box,0],new_bboxes[num_box,1]),(new_bboxes[num_box,2],new_bboxes[num_box,3]),(255,0,0),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,str(round(now_weight[num_box]/255.0*10000)/10000),(new_bboxes[num_box,0],new_bboxes[num_box,1]), font, 0.5,(0,0,255),1,cv2.LINE_AA)
                        attention_frame[int(new_bboxes[num_box,1]):int(new_bboxes[num_box,3]),int(new_bboxes[num_box,0]):int(new_bboxes[num_box,2])] = now_weight[num_box]

                    attention_frame = cv2.applyColorMap(attention_frame, cv2.COLORMAP_HOT)
                    dst = cv2.addWeighted(frame,0.6,attention_frame,0.4,0)
                    cv2.putText(dst,str(counter+1),(10,30), font, 1,(255,255,255),3)
                    cv2.imshow('result',dst)
                    c = cv2.waitKey(50)
                    ret, frame = cap.read()
                    if c == ord('q') and c == 27 and ret:
                        break;
                    counter += 1

            cv2.destroyAllWindows()

def test(model_path):
    # load model
    device = 'cpu'
    if torch.cuda.is_available():
        device='cuda'
    model = Model(device = device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("model restore!!!")
    print("Training")
    test_all(model,train_num,train_path,device)
    print("Testing")
    test_all(model,test_num,test_path,device)



if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.mode == 'train':
           train(args.resume)
    elif args.mode == 'test':
           test(args.model)
    elif args.mode == 'demo':
           vis(args.model)
