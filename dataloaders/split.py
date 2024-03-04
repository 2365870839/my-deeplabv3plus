import os
import numpy as np

root = r"F:\PAPERWORK\dataset\DeepCrack\JPEGImages"     # todo
output = r"F:\PAPERWORK\dataset\DeepCrack\ImageSets\Segmentation"     # todo
filename = []
# 从存放原图的目录中遍历所有图像文件
# dirs = os.listdir(root)
for root, dir, files in os.walk(root):
    for file in files:
        print(file)
        filename.append(file[:-4])  # 去除后缀，存储

# 打乱文件名列表
np.random.shuffle(filename)
# 划分训练集、测试集，默认比例6:2:2
train = filename[:int(len(filename) * 0.6)]
trainval = filename[int(len(filename) * 0.6):int(len(filename) * 0.8)]
val = filename[int(len(filename) * 0.8):]

# 分别写入train.txt, test.txt
with open(os.path.join(output, 'train.txt'), 'w') as f1, open(os.path.join(output, 'trainval.txt'), 'w') as f2, open(
        os.path.join(output, 'val.txt'), 'w') as f3:
    for i in train:
        f1.write(i + '\n')
    for i in trainval:
        f2.write(i + '\n')
    for i in val:
        f3.write(i + '\n')

print('成功！')