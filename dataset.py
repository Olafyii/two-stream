import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2 as cv
import random
import pickle
import sys
import datetime
# sys.path.insert(0, 'Z:\Im2Flow\two-stream')


def dump(a, path):
    f = open(path, 'wb')
    pickle.dump(a, f)
    f.close()
def load(path):
    f = open(path, 'rb')
    a = pickle.load(f)
    f.close()
    return a

# print(torch.cuda.is_available())
# for i in range(torch.cuda.device_count()):
#     print(torch.cuda.get_device_name(i))


class make_dataset():
    def __init__(self, data_root='Z:\\UCF101\\UCF-101'):
        self.data_root = data_root
        # self.len = sample_num
        self.all_data = {}  # key is index of a frame 0-2484199, value is a tuple (pathid, i_th frame, label)
        # path = self.flist[pathid]
        self.labelDict = {}  # key is name of directory, such as 'RopeClimbing', value is label
        self.flist = []
        for root, dirs, files in os.walk(self.data_root):
            if len(dirs) == 0:
                self.flist += [os.path.join(root, file) for file in files]
            else:
                for label, name in enumerate(dirs):
                    self.labelDict[name] = label
        self.labelDict['HandStandPushups'] = self.labelDict['HandstandPushups']  # name inconsistent in UCF101
        total_frame_num = 0
        for idx, f in enumerate(self.flist):
            # print(idx, len(self.all_data))
            video = cv.VideoCapture(f)
            frame_num = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            for frame_idx, i in enumerate(range(total_frame_num, total_frame_num+frame_num-10)):
                self.all_data[i] = (idx, frame_idx, self.labelDict[f.split('v_')[1].split('_g')[0]])
            total_frame_num += frame_num-10
        
        dump(self.flist, 'flist.list')
        dump(self.all_data, 'all_data.dict')
        dump(self.labelDict, 'labelDict.dict')
    
    def train_test_split(self, train_num=100000, test_num=50000):
        self.sampled_data_list = random.sample(range(len(self.all_data)), train_num+test_num)
        self.train_data = [self.all_data[i] for i in self.sampled_data_list[:train_num]]
        self.test_data = [self.all_data[i] for i in self.sampled_data_list[train_num:]]
        timestamp = datetime.datetime.now().strftime('%y.%m.%d.%H.%M.%S')
        dump(self.train_data, 'train_data_'+str(train_num)+'_'+timestamp+'.dict')
        dump(self.test_data, 'test_data_'+str(test_num)+'_'+timestamp+'.dict')
        

class UCF101(Dataset):
    def __init__(self, mode='train', sample_num=100000, loadName=''):
        self.flist = load('flist.list')
        self.labelDict = load('labelDict.dict')
        all_data = load('all_data.dict')
        if loadName != '':
            self.data = load(loadName)
            print('Loading train test split from %s' % loadName)
        else:
            sampled_data_list = random.sample(range(len(all_data)), sample_num)
            self.data = [all_data[i] for i in sampled_data_list]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        tup = self.data[index]
        img = cv.VideoCapture(self.flist[tup[0]])
        # print(self.flist[tup[0]], tup[1])
        img.set(cv.CAP_PROP_POS_FRAMES, tup[1])
        _, img = img.read()
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        # label = torch.zeros(101)
        # label[tup[2]] = 1
        label = tup[2]
        return img, label