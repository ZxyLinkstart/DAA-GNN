from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import cv2
import os
import time
import random
import pickle
import numpy as np
import gc
from extract.fasterrcnn import FasterRCNN
import imageio

try: # python2
    FRAME_WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
    FRAME_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
    FRAME_RATE = cv2.cv.CV_CAP_PROP_FPS
    FOURCC = cv2.cv.CV_FOURCC
except AttributeError: # python3
    FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
    FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
    FRAME_RATE = cv2.CAP_PROP_FPS
    FOURCC = cv2.VideoWriter_fourcc

class Extractor(object):
    def __init__(self, det_vis = False, write_video = False, pad_n = 20, dataset='DAD', out_path='output/dash'):
        self.vidcap = cv2.VideoCapture()
        # if det_vis:
        #     self.faster_rcnn = FasterRCNN(['--vis'])
        # else:
        #     self.faster_rcnn = FasterRCNN()
        self.faster_rcnn = FasterRCNN(dataset=dataset, det_vis=det_vis)
        self.class_names = self.faster_rcnn.class_names
        self.write_video = write_video
        self.det_output=None
        self.pad_n = pad_n
        self.out_path = out_path
    
    def open(self, video_path):
        self.vidcap.open(video_path)
        self.im_width = int(self.vidcap.get(FRAME_WIDTH))
        self.im_height = int(self.vidcap.get(FRAME_HEIGHT))
        self.frame_rate = int(self.vidcap.get(FRAME_RATE))
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            video_id = os.path.basename(video_path)[:-4]
            self.det_output = imageio.get_writer("{:s}/{:s}.mp4".format(self.out_path, video_id), fps=20)
        return self.vidcap.isOpened()
    
    def extract_and_pad(self):
        all_frame_feat = []
        all_obj_feat = []
        all_det = []
        num_obj = []
        while self.vidcap.grab():
            # start = time.time()
            # BGR image
            _, ori_im = self.vidcap.retrieve()
            # [frame_feat, obj1_feat, ...],[(box_xyxy, conf, id),...]
            res_feat, res_dect = self.faster_rcnn(ori_im, self.det_output, use_nms=True)
            # res_dect = res_dect.cpu().numpy()
            # res_feat = res_feat.cpu().numpy()
            # print('\t',len(res_dect)-1,"objects")
            all_frame_feat.append(res_feat[0,:])
            if len(res_dect)-1 < self.pad_n:
                pad = np.zeros((self.pad_n-len(res_feat)+1,len(res_feat[0])))
                padded_obj_feat = np.vstack((res_feat[1:,],pad))
                all_obj_feat.append(padded_obj_feat)
                pad_det = np.zeros((self.pad_n-len(res_dect)+1,len(res_dect[0])))
                padded_det = np.vstack((res_dect[1:,],pad_det))
                all_det.append(padded_det)
                num_obj.append(len(res_dect)-1)
            else:
                all_obj_feat.append(res_feat[1:self.pad_n+1,])
                all_det.append(res_dect[1:self.pad_n+1,])
                num_obj.append(self.pad_n)
            # end = time.time()
        self.det_output.close()
        all_det = np.stack(all_det,axis=0).astype(np.float32)
        num_obj = np.stack(num_obj,axis=0)
        all_frame_feat = np.stack(all_frame_feat,axis=0).astype(np.float32)
        all_obj_feat = np.stack(all_obj_feat,axis=0).astype(np.float32)
        # t v c -> c t v m
        # data = np.transpose(data,(0,3,2,1))
        # data = np.expand_dims(data,axis=-1)
        # all_obj_feat = np.transpose(all_obj_feat, (2,0,1))
        # all_obj_feat = np.expand_dims(all_obj_feat, axis = -1)
        return num_obj, all_det, all_obj_feat, all_frame_feat 
    

if __name__ == "__main__":
    extractor = Extractor(det_vis=False, write_video=True)
    _root = 'data/videos/'
    _pos, _neg = 'positive', 'negative'
    _train, _test = 'training', 'testing'
    _pos_label = 1
    _neg_label = 0
    out_path = 'data/tmp/'
    for _split in [_train, _test]:
        _split_path = os.path.join(out_path,_split)
        if not os.path.exists(_split_path):
            os.makedirs(_split_path)
        print("handling {} data".format(_split))
        pos_list = [os.path.join(_root,_split,_pos, x) for x in os.listdir(os.path.join(_root,_split,_pos))]
        neg_list = [os.path.join(_root,_split,_neg, x) for x in os.listdir(os.path.join(_root,_split,_neg))]
        all_name = pos_list + neg_list

        for sample in all_name:
            ext_tic = time.time()
            print("extracting video {}".format(sample))
            extractor.open(sample)
            num_obj, det, feat, ffeat = extractor.extract_and_pad()
            ext_toc = time.time()
            # n v t c -> n c t v m
            # data = np.transpose(data,(0,3,2,1))
            # data = np.expand_dims(data,axis=-1)
            print('\tdone. time: {:.2f} s'.format(ext_toc-ext_tic))
            file_name = '_'.join(sample.split('/')[-2:])[:-4]
            # np.savez('{}/{}/{}.npz'.format(out_path,_split, file_name), num_obj=num_obj,det=det, feat=feat, ffeat=ffeat)
