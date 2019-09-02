import os
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils, models
from torch.autograd import Variable
from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from tqdm import tqdm
import time

from dataset_rgb_flow import UCF101flow

if __name__ == '__main__':  # multithread loading needs to specify this.
    trainset = UCF101flow(mode='train')
    trainloader = DataLoader(trainset, batch_size=256, num_workers=0)
    valset = UCF101flow(mode='val')
    valloader = DataLoader(valset, batch_size=256, num_workers=0)

    alexnet = models.alexnet(pretrained=True).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(alexnet.parameters(), 1e-5, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1,verbose=True)

    for epoch in range(10):
        progress = tqdm(trainloader)
        t = time.time()
        for i, (data, label) in enumerate(progress):
            print('load time:', time.time() - t)
            t = time.time()
            label = label.cuda(async=True)
            input_var = Variable(data).cuda()
            target_var = Variable(label).cuda()
            print('load to cuda:', time.time() - t)

            t = time.time()
            output = alexnet(input_var)
            loss = criterion(output, target_var)
            # print('loss:', loss.item())
            print('calculate loss:', time.time() - t)

            t = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('backward:', time.time() - t)
            t = time.time()
