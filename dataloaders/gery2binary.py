import os
from PIL import Image
import numpy as np

folder_path = r"F:\PAPERWORK\dataset\DeepCrack\SegmentationClass1"  # 图片路径文件夹
save_path = r"F:\PAPERWORK\dataset\DeepCrack\SegmentationClass"  # 保存路径文件夹

file_list = os.listdir(folder_path)
for img in file_list:
    img_path = folder_path + '\\' + img
    src = Image.open(img_path)
    src_arr = np.array(src)  # debug用
    binary = src.convert('1')
    binary_arr = np.array(binary)  # debug用
    binary.save(save_path + '\\' + img[:-4] + '.png')
