import os
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils
import pickle
from PIL import Image
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class UCF101flow(Dataset):
    def __init__(self, mode='train'):
        print('test')
        if mode == 'train':
            f = open('./train_test_split/500000.txt')
            self.transform = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        elif mode == 'val':
            f = open('./train_test_split/50000.txt')
            self.transform = transforms.Compose([
                transforms.Resize([256,256]),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        else:
            print('ilegal model')
            return
        self.paths = f.readlines()
        f.close()

        f = open('./train_test_split/label.dict', 'rb')
        self.dict = pickle.load(f)
        f.close()
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx][:-1]
        img = Image.open(path)
        # print(path)
        # print(img.size)
        img = self.transform(img)
        label = self.dict[path.split('v_')[1].split('_g')[0]]
        return img, label

trainset = UCF101flow()
y = []
total_t = 0
for i in range(0, len(trainset)):
    t = time.time()
    trainset[i]
    total_t += (time.time() - t)
    t = time.time()
    if i % 100 == 99:
        print(i, total_t)
        # y.append(total_t)
        total_t = 0
f = open('opentime.list', 'wb')
pickle.dump(y, f)
f.close()

fig, ax = plt.subplots()
x = np.arange(len(y))
x = x*100
ax.plot(x, y)
plt.show()