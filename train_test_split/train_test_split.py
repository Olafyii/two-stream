import os

data_root='D:\\data\\UCF101_RGB_flow\\jpegs_256'

# 生成4个txt，rgb的train，test，flow的train，test，要对应的
# （其实不用flow的，flow是自己生成的）

all_txt = open('./train_test_split/all.txt', 'w')
i = 0
for root, dirs, files in os.walk(data_root):
    if len(dirs) == 0:
        for file in files:
            all_txt.write(root+'\\'+file+'\n')
            print(i)
            i += 1
all_txt.close()