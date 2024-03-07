from dataloaders.datasets import deepcrack
import argparse
from modeling.deeplab import DeepLab
import torch
from torch.utils.data import DataLoader
import numpy as np
from utils.metrics import Evaluator
import os

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.base_size = 513
args.crop_size = 513
args.nclass = 2
args.backbone = "resnet"
args.out_stride = 16
args.model_path = "model_best.pth.tar"
args.batch_size = 4
args.save_dir = "run/deepcrack/deeplab-resnet/experiment_0"

test_sets = deepcrack.DeepCrack(args, split='test')
test_loader = DataLoader(test_sets, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
model = DeepLab(num_classes=args.nclass,
                backbone=args.backbone,
                output_stride=args.out_stride,
                sync_bn=False,
                freeze_bn=False)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.to('cuda')
model.eval()
evaluator = Evaluator(args.nclass)
for ii, sample in enumerate(test_loader):
    image, target = sample['image'], sample['label']
    image, target = image.cuda(), target.cuda()
    with torch.no_grad():
        output = model(image)
    pred = output.data.cpu().numpy()
    target = target.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    # Add batch sample into evaluator
    evaluator.add_batch(target, pred)

# Fast test during the training
Acc = evaluator.Pixel_Accuracy()
Acc_class = evaluator.Pixel_Accuracy_Class()
mIoU = evaluator.Mean_Intersection_over_Union()
FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
Dice = evaluator.Dice_score()
Precision, Recall, F1score = evaluator.precision_recall()
print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, Dice: {}, Precision: {}, Recall: {}, F1score: {}".format(Acc, Acc_class, mIoU, FWIoU, Dice, Precision, Recall,F1score))
with open(os.path.join(args.save_dir,'test_result.txt'), 'w') as f:
    f.write("Acc:{}\n, Acc_class:{}\n, mIoU:{}\n, fwIoU: {}\n, Dice: {}\n, Precision: {}\n, Recall: {}\n, "
            "F1score: {}\n".format(Acc, Acc_class, mIoU, FWIoU, Dice, Precision, Recall, F1score))

