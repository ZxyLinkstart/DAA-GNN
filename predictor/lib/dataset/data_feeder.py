from __future__ import print_function
from __future__ import division
import os
import pickle
import numpy as np
from torch.utils.data import Dataset

class DataFeeder(Dataset):
    def __init__(self, data_path, box_num=None):
        self._pos_prefix = 'positive'
        self.data_path = data_path
        self.load_data(data_path)
        self.box_num = box_num
        
    def load_data(self, data_path):
        self.names = os.listdir(data_path)
        self.labels = [ int(name.startswith(self._pos_prefix)) \
            for name in self.names]
        # self.N = len(self.names)
        # _,feat,_,_ = self.__getitem__(0)
        # self.C, self.T, self.V, self.M = feat.shape

    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.names[index])
        data = np.load(path, allow_pickle=True, encoding='bytes')
        num_obj, det, feat, ffeat = data['num_obj'],data['det'],data['feat'],data['ffeat']
        # det, feat, ffeat = data[0],data[1],data[2]
        label = self.labels[index]
        return self.names[index], num_obj, det, feat, ffeat, label
        # return det, feat, ffeat, label


if __name__ == "__main__":
    dataset = DataFeeder('data/ef_proposal_0113/training')
    N = 20
    for name, num_obj, det, feat, ffeat, label in dataset:
        conf = det[:,:,4]
        idx = np.argsort(conf, axis=-1)
        idx = idx[:,::-1]
        det = np.array([det[i][idx[i]] for i in range(len(det))])
        feat = np.array([feat[i][idx[i]] for i in range(len(feat))])
        conf = det[:,:,4]
        num_obj.fill(N)
        det = det[:,:N]
        feat = feat[:,:N]
        np.savez('data/ef_proposal_0113_20p/training/{}'.format(name), num_obj=num_obj,det=det, feat=feat, ffeat=ffeat)