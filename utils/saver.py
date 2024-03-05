import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        # 匹配路径 run/dataset_name/deeplab-resnet/experiment_*
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))     # glob.glob获取所有匹配的路径，组成一个列表
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0      # 最后一个experiment的id + 1

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))     # 保存的路径: run/dataset_name/deeplab-resnet/experiment_id
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)        # 如果要保存的路径不存在就先创建

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)  # 保存checkpoint的地址: run/dataset_name/deeplab-resnet/experiment_id/checkpoint.pth.tar
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')       # 保存超参数log的路径
        log_file = open(logfile, 'w')
        p = OrderedDict()       # 有序字典，用来记录超参数
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():      # 遍历字典，写入log文件中
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()