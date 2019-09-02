import os
import time
import random
# t = time.time()
# f = open('all.txt')
# imgs = f.readlines()
# f.close()
# print(time.time()-t)
# imgs = random.sample(imgs, 500000)
# f = open('500000.txt', 'w')
# for i in imgs:
#     f.write(i)
# f.close()
f = open('500000.txt')
imgs = f.readlines()
f.close()
for idx, i in enumerate(imgs):
    print(idx)
    i = i[:-1]
    os.system('echo f | xcopy '+i+' D:\\data\\UCF101_RGB_flow\\500000\\'+'_'.join(i.split('\\')[-2:]))