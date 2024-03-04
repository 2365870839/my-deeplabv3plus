from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


class DeepCrack(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 2  # todo

    def __init__(self,
                 args,
                 # todo
                 base_dir=Path.db_root_dir('deepcrack'),  # path_to_dataset_folder/
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')  # path_to_dataset_folder/JPEGImages
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')  # path_to_dataset_folder/SegmentationClass

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets',
                                   'Segmentation')  # path_to_dataset_folder/ImageSets/Segmentation

        self.im_ids = []  # 原图的文件名列表
        self.images = []  # 原图的路径列表
        self.categories = []  # 标签图的路径列表

        for splt in self.split:  # splt是train或val
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')),
                      "r") as f:  # path_to_dataset_folder/ImageSets/Segmentation/train.txt
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir,
                                      line + ".jpg")  # 原图路径（jpg格式）：path_to_dataset_folder/JPEGImages/xxx.jpg
                _cat = os.path.join(self._cat_dir,
                                    line + ".png")  # 标签路径（png格式）：path_to_dataset_folder/JPEGImages/xxx.png
                assert os.path.isfile(_image)  # 判断是不是已经存在的文件
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))  # 判断原图数量是不是和标签数量一致

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):  # 在对该类的对象使用len方法时会调用该函数
        return len(self.images)

    def __getitem__(self, index):  # 在对该类的对象使用索引时会调用该函数
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)  # 训练数据处理，返回预处理+数据增强后的结果,返回字典，字典里的value由PIL转为了Tensor
            elif split == 'val':
                return self.transform_val(sample)  # 验证数据处理，返回预处理的结果，不做数据增强，返回字典，字典里的value由PIL转为了Tensor

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')  # 返回RGB格式的图片文件（一个Image对象）
        _target = Image.open(self.categories[index])  # 返回标签的图片文件

        return _img, _target

    # 训练数据预处理+数据增强
    def transform_tr(self, sample):
        # 调用了custom_transforms中的图像处理函数/类，生成匿名对象
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        # 调用transforms.Compose的__call__方法：调用composed_transforms中的每个匿名对象
        return composed_transforms(sample)

    # 验证数据预处理
    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):  # 对该类对象使用str和print方法时调用该函数，str方法会返回该函数的返回值，print方法直接打印该函数的返回值
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = DeepCrack(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        # 因为数据集类实现了gititem方法，所以这里的sample就是gititem方法的返回值，即做过预处理的存储在字典中的Tensor类型的原图和mask数据
        # 因为创建Dataloader时设置batchsize为5，所以每次取5组原图+标签，合并到一起（batchsize维度上合并），在放入sample中
        for jj in range(sample["image"].size()[0]): # batchsize维度
            img = sample['image'].numpy()   # ndarray类型原图
            gt = sample['label'].numpy()    # ndarray类型标签
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='deepcrack')   # mask调色板
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 0:
            break

    plt.show(block=True)
