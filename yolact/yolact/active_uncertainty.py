import pickle
from tqdm import tqdm
from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from yolact import Yolact
from torch.utils.tensorboard import SummaryWriter
from data.local import tb_dir
import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime

# Oof
import eval as eval_script
from data.local import tb_dir
from extract_sub_dataset import extract_sub_dataset

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

from train import train
import json
from data.local import tmp_json_root, weight_root
from train import get_loss
from data.local import coco_mini_person_train

torch.manual_seed(61) # 为CPU设置随机种子
torch.cuda.manual_seed(61) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(61)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(61)  # Numpy module.
random.seed(61)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--init_ratio', default=0.1, type=float,
                    help='利用多少个比例的数据初始化训练网络')
parser.add_argument('--ratio_list', default=None, type=str,
                    help='主动学习设置重后期每轮的筛选样本比例')
parser.add_argument('--exp_setting', default=None, type=int,
                    help='筛除掉大损失还是小损失')
parser.add_argument('--loss_path', default=None, type=str,
                    help='损失值的路径')
parser.add_argument('--ratio', default=None, type=float,
                    help='抛弃多少比例高损失样本')
parser.add_argument('--exp_name', default=None, type=str,
                    help='name of experiment')
parser.add_argument('--num_epochs', default=80, type=int,
                    help='number of epoch for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"' \
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be' \
                         'determined from the file name.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default=weight_root,
                    help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='logs/',
                    help='Directory for saving logs.')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--save_interval', default=10000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--save_epochs', default=1, type=int,
                    help='The number of iterations between saving the model.')

parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=1, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)

args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]


# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))


replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if args.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net: Yolact, criterion: MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses


#  这里这个模块用来计算每个样本的损失值
class NetLossSingle(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net: Yolact, criterion: MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion.forward_single(self.net, preds, targets, masks, num_crowds)
        return losses


#  这里这个模块用来计算每个样本的置信度
class NetConfidenceSingle(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net: Yolact, criterion: MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        confidence = self.criterion.get_confidence(self.net, preds, targets, masks, num_crowds)
        return confidence


class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
               [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])

        return out


#  这里的 PseudoDataParallel实际上对outputs没有做任何操作 net加上PseudoDataParallel只是为了通过hook
class PseudoDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
               [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        return outputs


class ScatterWrapper:
    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, list):
                print("Warning: ScatterWrapper got input of non-list type")
        self.args = args
        self.batch_size = len(args[0])

    def make_mask(self):
        out = torch.Tensor(list(range(self.batch_size))).long()
        if args.cuda:
            return out.cuda()
        else:
            return out

    def get_args(self, mask):
        device = mask.device
        mask = [int(x) for x in mask]
        out_args = [[] for _ in self.args]

        for out, arg in zip(out_args, self.args):
            for idx in mask:
                x = arg[idx]
                if isinstance(x, torch.Tensor):
                    x = x.to(device)
                out.append(x)
        return out_args


def eval():
    if args.exp_name is None:
        print("请输入实验名称")
        return
    tensorboard_dir = os.path.join(tb_dir, args.exp_name)
    writer = SummaryWriter(tensorboard_dir)
    save_folder = os.path.join(args.save_folder, args.exp_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))

    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    net = yolact_net
    net.train()
    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
                  overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        # if args.start_iter == -1:
        #     args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio,
                             )

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:
        net = net.cuda()

    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn()  # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    compute_validation_map(0, 0, yolact_net, val_dataset, writer, log if args.log else None)


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    global cur_lr
    cur_lr = new_lr


def gradinator(x):
    x.requires_grad = False
    return x


def prepare_data(datum, devices: list = None, allocation: list = None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation))  # The rest might need more/less

        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        if len(images) != args.batch_size:
            debug = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx] = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx] = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images) - 1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)

        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx] = torch.stack(images[cur_idx:cur_idx + alloc], dim=0)
            split_targets[device_idx] = targets[cur_idx:cur_idx + alloc]
            split_masks[device_idx] = masks[cur_idx:cur_idx + alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx + alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds


def no_inf_mean(x: torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()


def compute_validation_loss(net, data_loader, criterion):
    global loss_types

    with torch.no_grad():
        losses = {}

        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())

            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break

        for k in losses:
            losses[k] /= iterations

        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)


#  这里利用tensorboard记录数据
def compute_validation_map(epoch, iteration, yolact_net, dataset, writer, log: Log = None):
    with torch.no_grad():
        yolact_net.eval()

        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()
        for key in val_info['box'].keys():
            writer.add_scalar('box_ap' + str(key), val_info['box'][key], iteration)

        for key in val_info['mask'].keys():
            writer.add_scalar('mask_ap' + str(key), val_info['mask'][key], iteration)

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()


#  创建一个根据list构造dataset的函数
def build_dataset_from_list(exp_name, image_file_name_list):
    # 构造一个临时的sub_json文件
    with open(cfg.dataset.train_info, 'r', encoding='utf8') as fp:
        info_json = json.load(fp)
    selected_images = []
    selected_image_ids = []
    print("开始筛选images")
    for image in tqdm(info_json['images']):
        if image['file_name'] in image_file_name_list:
            selected_images.append(image)
            selected_image_ids.append(image['id'])

    print("开始筛选annotations")
    selected_annotations = []
    for annotation in tqdm(info_json['annotations']):
        if annotation['image_id'] in selected_image_ids:
            selected_annotations.append(annotation)

    info_json['annotations'] = selected_annotations
    info_json['images'] = selected_images

    json_path = os.path.join(tmp_json_root, exp_name + '_info_json.json')
    if os.path.exists(json_path):
        os.remove(json_path)

    if not os.path.exists(tmp_json_root):
        os.makedirs(tmp_json_root)
    print("开始暂存json文件")
    # json.dump(info_json, open(json_path, "w"))
    jsObj = json.dumps(info_json)
    with open(json_path, "w") as f:
        f.write(jsObj)
        f.close()
    print("暂存json文件完毕")

    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=json_path,
                            transform=SSDAugmentation(MEANS))
    return dataset
    debug = 0


def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images=' + str(args.validation_size)])


#  主动学习中uncertainty的实验设置 优先挑损失小的样本和随机挑选样本进行比对
# TODO 添加参数
def active_uncertainty(init_ratio=None,
                       num_epochs=None,
                       exp_setting=None,
                       ratio_list=None):
    """

    :param init_ratio:
    :param num_epochs:
    :param exp_setting:
    :param ratio_list:
    :return:
    """
    if args.init_ratio is not None:
        init_ratio = args.init_ratio
    if init_ratio is None:
        print("请输入参数：drop_ratios")
        return

    if args.num_epochs is not None:
        num_epochs = args.num_epochs
    if num_epochs is None:
        print("请输入参数num_epochs")
        return

    if args.exp_setting is not None:
        exp_setting = args.exp_setting
    if exp_setting == 0:
        exp_setting = 'drop_small_loss'
    elif exp_setting == 1:
        exp_setting = 'drop_large_loss'
    else:
        print("请输入正确的实验设置参数 exp_setting")
        return

    if args.ratio_list is not None:
        ratio_list = args.ratio_list.split(",")
        ratio_list = [float(x) for x in ratio_list]
    if ratio_list is None:
        print("请输入参数：drop_ratios")
        return

    #  uncertainty 和 random 有一个已经加入到训练集的list
    random_selected_list = []
    uncertainty_selected_list = []

    json_path = cfg.dataset.train_info
    init_json = os.path.join(tmp_json_root, cfg.dataset.name + '_{}.json'.format(init_ratio))
    tmp = extract_sub_dataset(ann_file=json_path,
                              extract_num=init_ratio,
                              new_ann_file=init_json)
    random_selected_list.extend(tmp)
    uncertainty_selected_list.extend(tmp)

    init_dataset = COCODetection(image_path=cfg.dataset.train_images,
                                 info_file=init_json,
                                 transform=SSDAugmentation(MEANS))
    init_exp_name = 'train_{}_dataset_{}_epochs_with_{}_samples'.format(cfg.dataset.name, num_epochs, init_ratio)
    train(exp_name=init_exp_name,
          dataset=init_dataset,
          num_epochs=num_epochs)

    #  每次增量%2, 训练80个epoch

    #  先根据损失筛选样本 再随机筛选样本
    for i, ratio in enumerate(ratio_list):
        #  根据上一轮的模型对样本进行损失值预测， 再根据损失值挑选样本

        if i == 0:
            last_step_uncertainty_exp_name = init_exp_name
        else:
            last_step_uncertainty_exp_name = 'train_{}_active_uncertainty_{}_with_{}_samples'.format(cfg.dataset.name, exp_setting, ratio_list[i-1])

        last_step_uncertainty_save_folder = os.path.join(args.save_folder, last_step_uncertainty_exp_name)

        last_step_best_uncertainty_weights_path = os.path.join(last_step_uncertainty_save_folder, 'best_mean_ap.pth')

        loss_list = get_loss(last_step_best_uncertainty_weights_path)

        if exp_setting == 'drop_large_loss':
            loss_list.sort(key=lambda x: x['loss'])
        else:
            loss_list.sort(key=lambda x: x['loss'], reverse=True)

        all_image_list = [x['file_name'] for x in loss_list]
        #  按顺序筛选出样本加入uncertainty_selected_list, 直到uncertainty_selected_list的长度达到预期，注意去重
        target_len = math.ceil(ratio * len(loss_list))
        for i, image in enumerate(all_image_list):
            if len(uncertainty_selected_list) > target_len:
                break
            if image not in uncertainty_selected_list:
                uncertainty_selected_list.append(image)

        uncertainty_exp_name = 'train_{}_active_uncertainty_{}_with_{}_samples'.format(cfg.dataset.name, exp_setting, ratio)
        random_exp_name = 'train_{}_random_with_{}_samples'.format(cfg.dataset.name, ratio)

        uncertainty_dataset = build_dataset_from_list(exp_name=uncertainty_exp_name,
                                                      image_file_name_list=uncertainty_selected_list)
        train(exp_name=uncertainty_exp_name,
              dataset=uncertainty_dataset,
              num_epochs=num_epochs,
              )

        random.shuffle(all_image_list)
        for i, image in enumerate(all_image_list):
            if len(random_selected_list) > target_len:
                break
            if image not in random_selected_list:
                random_selected_list.append(image)

        random_dataset = build_dataset_from_list(exp_name=random_exp_name,
                                                 image_file_name_list=random_selected_list)
        train(exp_name=random_exp_name,
              dataset=random_dataset,
              num_epochs=num_epochs,
              )


if __name__ == '__main__':
    active_uncertainty()
