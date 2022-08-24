import pickle

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
from tqdm import tqdm


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


import json
from data.local import tmp_json_root, weight_root, confidence_root, score_root

torch.manual_seed(61)  # 为CPU设置随机种子
torch.cuda.manual_seed(61)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(61)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(61)  # Numpy module.
random.seed(61)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from train import train
parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--exp_name', default=None, type=str,
                    help='name of experiment')
parser.add_argument('--confidence_path', default=None, type=str,
                    help='置信度值list路径')
parser.add_argument('--confidence_threshold', default=None, type=float,
                    help='置信度值阈值')
parser.add_argument('--exp_setting', default=None, type=int,
                    help='筛除掉大损失还是小损失')
parser.add_argument('--weight_to_get_loss_path', default=None, type=str,
                    help='获取损失值的权重路径')
parser.add_argument('--drop_ratios', default=None, type=str,
                    help='每轮筛除掉的样本比例')
parser.add_argument('--step_ratios', default=None, type=str,
                    help='两阶段训练的 训练轮数比例')
parser.add_argument('--num_epochs', default=50, type=int,
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

    def __init__(self, net: Yolact, criterion: MultiBoxLoss, threshold):
        super().__init__()

        self.net = net
        self.criterion = criterion
        self.threshold = threshold

    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        confidence = self.criterion.get_confidence(self.net, preds, targets, masks, num_crowds, self.threshold)
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
# TODO 把devices 和 allocation固定
class PseudoDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:0']
        splits = prepare_data(inputs[0], devices, allocation=[1])

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

def get_latest_epoch_iteration(name='yolact_base_82_459239.pth'):
    lst = name.split('_')
    iteration = lst[-1].split('.')[-2]
    epoch = lst[-2]
    return int(epoch), int(iteration)


def train_step(exp_name=None, num_epochs=None, dataset=None, resume=None):
    if exp_name is None:
        exp_name = args.exp_name
    if exp_name is None:
        print("请输入实验名称")
        return
    if resume is not None:
        args.resume = resume
    # 根据exp_name保存对应的数据
    tensorboard_dir = os.path.join(tb_dir, exp_name)
    writer = SummaryWriter(tensorboard_dir)
    save_folder = os.path.join(args.save_folder, exp_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if dataset is None:
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
        args.resume = SavePath.get_interrupt(save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(save_folder, cfg.name)
        file_name = os.path.basename(args.resume)
        start_epoch, star_iteration = get_latest_epoch_iteration(file_name)

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

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    # 一个epoch中需要多少个iteration
    epoch_size = len(dataset) // args.batch_size
    # num_epochs = math.ceil(cfg.max_iter / epoch_size)
    if num_epochs is None:
        num_epochs = args.num_epochs
    cfg.max_iter = math.ceil(num_epochs * epoch_size)
    # 按iteration 在 cfg.max_iter中的比例设置learning rate
    cfg.lr_steps = [math.ceil(0.35 * cfg.max_iter), math.ceil(0.75 * cfg.max_iter), math.ceil(0.875 * cfg.max_iter),
                    math.ceil(0.975 * cfg.max_iter)]

    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)

    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch + start_epoch, iteration + star_iteration).get_path(root=save_folder)
    time_avg = MovingAverage()
    global loss_types  # Forms the print order
    loss_avgs = {k: MovingAverage(100) for k in loss_types}
    print('Begin training!')
    # try-except so you can use ctrl+c to save early and stop training
    # TODO 保存训练过程中指标最好的模型权重和
    best_box_ap = 0
    best_box_ap_path = os.path.join(save_folder, 'best_box_ap.pth')
    best_mask_ap = 0
    best_mask_ap_path = os.path.join(save_folder, 'best_mask_ap.pth')
    best_mean_ap = 0
    best_mean_ap_path = os.path.join(save_folder, 'best_mean_ap.pth')
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch + 1) * epoch_size < iteration:
                continue
            for datum in data_loader:

                # Stop at the configured number of iterations even if mid-epoch
                # if iteration == cfg.max_iter:
                #     break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])
                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer,
                           (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration

                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))

                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)

                losses = {k: (v).mean() for k, v in losses.items()}  # Mean here because Dataparallel

                loss = sum([losses[k] for k in losses])

                writer.add_scalar('train loss', loss, iteration)

                writer.add_scalar('lr in optimizer', optimizer.param_groups[0]['lr'], iteration)
                #  利用tensorboard记录损失
                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])
                # Backprop
                loss.backward()  # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time = time.time()
                elapsed = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = \
                    str(datetime.timedelta(seconds=(cfg.max_iter - iteration) * time_avg.get_avg())).split('.')[0]

                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                          % tuple([epoch + start_epoch, iteration + star_iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(loss.item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0)  # nvidia-smi is sloooow

                    log.log('train', loss=loss_info, epoch=epoch + start_epoch, iter=iteration + star_iteration,
                            lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu

                iteration += 1

            # This is done per epoch 每个epoch将模型保存下来
            yolact_net.save_weights(save_path(epoch + start_epoch, iteration + star_iteration))
            box_ap, mask_ap = compute_validation_map(epoch + start_epoch, iteration + star_iteration, yolact_net, val_dataset, writer,
                                                     log if args.log else None)
            mean_ap = (box_ap + mask_ap) / 2

            if box_ap > best_box_ap:
                best_box_ap = box_ap
                if os.path.exists(best_box_ap_path):
                    os.remove(best_box_ap_path)
                yolact_net.save_weights(best_box_ap_path)
                writer.add_scalar('best_box_ap', best_box_ap, 1)

            if mask_ap > best_mask_ap:
                best_mask_ap = mask_ap
                if os.path.exists(best_mask_ap_path):
                    os.remove(best_mask_ap_path)
                yolact_net.save_weights(best_mask_ap_path)
                writer.add_scalar('best_mask_ap', best_mask_ap, 1)

            if mean_ap > best_mean_ap:
                best_mean_ap = mean_ap
                if os.path.exists(best_mean_ap_path):
                    os.remove(best_mean_ap_path)
                yolact_net.save_weights(best_mean_ap_path)
                writer.add_scalar('best_mean_ap', best_mean_ap, 1)

        # Compute validation mAP after training is finished
        compute_validation_map(epoch + start_epoch, iteration + star_iteration, yolact_net, val_dataset, writer, log if args.log else None)

    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(save_folder)
            yolact_net.save_weights(save_path(epoch + start_epoch, repr(iteration + star_iteration) + '_interrupt'))
        exit()

    yolact_net.save_weights(save_path(epoch + start_epoch, iteration + star_iteration))


def train_step2(exp_name=None, num_epochs=None, dataset=None, resume=None):
    if exp_name is None:
        exp_name = args.exp_name
    if exp_name is None:
        print("请输入实验名称")
        return
    if resume is not None:
        args.resume = resume
    # 因为是第二阶段，所以我们要从第一阶段中的保存路径中取出文件，所以构造resume_folder
    resume_folder = os.path.join(args.save_folder, exp_name)
    # 第二阶段和第一阶段的保存路径分开
    exp_name = exp_name + '_step2'
    tensorboard_dir = os.path.join(tb_dir, exp_name)
    writer = SummaryWriter(tensorboard_dir)
    save_folder = os.path.join(args.save_folder, exp_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if dataset is None:
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
        args.resume = SavePath.get_interrupt(resume_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(resume_folder, cfg.name)

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

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    # 一个epoch中需要多少个iteration
    epoch_size = len(dataset) // args.batch_size
    # num_epochs = math.ceil(cfg.max_iter / epoch_size)
    if num_epochs is None:
        num_epochs = args.num_epochs
    cfg.max_iter = math.ceil(num_epochs * epoch_size)
    # 按iteration 在 cfg.max_iter中的比例设置learning rate
    cfg.lr_steps = [math.ceil(0.35 * cfg.max_iter), math.ceil(0.75 * cfg.max_iter), math.ceil(0.875 * cfg.max_iter),
                    math.ceil(0.975 * cfg.max_iter)]

    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)

    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=save_folder)
    time_avg = MovingAverage()
    global loss_types  # Forms the print order
    loss_avgs = {k: MovingAverage(100) for k in loss_types}
    print('Begin training!')
    # try-except so you can use ctrl+c to save early and stop training
    # TODO 保存训练过程中指标最好的模型权重和
    best_box_ap = 0
    best_box_ap_path = os.path.join(save_folder, 'best_box_ap.pth')
    best_mask_ap = 0
    best_mask_ap_path = os.path.join(save_folder, 'best_mask_ap.pth')
    best_mean_ap = 0
    best_mean_ap_path = os.path.join(save_folder, 'best_mean_ap.pth')
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch + 1) * epoch_size < iteration:
                continue
            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch + 1) * epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])
                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer,
                           (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration

                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))

                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)

                losses = {k: (v).mean() for k, v in losses.items()}  # Mean here because Dataparallel

                loss = sum([losses[k] for k in losses])

                writer.add_scalar('train loss', loss, iteration)

                writer.add_scalar('lr in optimizer', optimizer.param_groups[0]['lr'], iteration)
                #  利用tensorboard记录损失
                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])
                # Backprop
                loss.backward()  # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time = time.time()
                elapsed = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = \
                    str(datetime.timedelta(seconds=(cfg.max_iter - iteration) * time_avg.get_avg())).split('.')[0]

                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                          % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(loss.item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0)  # nvidia-smi is sloooow

                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                            lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu

                iteration += 1

            # This is done per epoch 每个epoch将模型保存下来
            yolact_net.save_weights(save_path(epoch, iteration))
            box_ap, mask_ap = compute_validation_map(epoch, iteration, yolact_net, val_dataset, writer,
                                                     log if args.log else None)
            mean_ap = (box_ap + mask_ap) / 2

            if box_ap > best_box_ap:
                best_box_ap = box_ap
                if os.path.exists(best_box_ap_path):
                    os.remove(best_box_ap_path)
                yolact_net.save_weights(best_box_ap_path)
                writer.add_scalar('best_box_ap', best_box_ap, 1)

            if mask_ap > best_mask_ap:
                best_mask_ap = mask_ap
                if os.path.exists(best_mask_ap_path):
                    os.remove(best_mask_ap_path)
                yolact_net.save_weights(best_mask_ap_path)
                writer.add_scalar('best_mask_ap', best_mask_ap, 1)

            if mean_ap > best_mean_ap:
                best_mean_ap = mean_ap
                if os.path.exists(best_mean_ap_path):
                    os.remove(best_mean_ap_path)
                yolact_net.save_weights(best_mean_ap_path)
                writer.add_scalar('best_mean_ap', best_mean_ap, 1)

        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, yolact_net, val_dataset, writer, log if args.log else None)

    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(save_folder)
            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    yolact_net.save_weights(save_path(epoch, iteration))

# TODO 新增一个能根据权重返回网络的函数
def get_net(resume=None):
    yolact_net = Yolact()
    net = yolact_net
    net.train()
    if resume is None:
        resume = args.resume
    if resume is None:
        print("请指定加载的网络模型")
    yolact_net.load_weights(resume)

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

    return yolact_net, net


def train_2_stage(exp_name, num_epochs=100, step_ratio=0.25):
    stage1_num_epochs = math.ceil(num_epochs * step_ratio)
    stage2_num_epochs = num_epochs - stage1_num_epochs
    train(exp_name, num_epochs=stage1_num_epochs)

    score_list_path = os.path.join(score_root, 'yolact_score.pickle')
    with open(score_list_path, 'rb') as f:
        score_list = pickle.load(f)
    score_list.sort(key=lambda x: x['score'], reverse=True)
    all_image_list = [x['file_name'] for x in score_list]
    selected_list = all_image_list[:math.ceil(len(all_image_list) * (1 - 0.2))]

    dataset = build_dataset_from_list(exp_name='tmp',
                                      image_file_name_list=selected_list)
    train_step2(exp_name, num_epochs=stage2_num_epochs, dataset=dataset, resume='latest')


def train_step_by_step(step_ratios=None):
    if args.step_ratios is not None:
        step_ratios = args.step_ratios.split(",")
        step_ratios = [float(x) for x in step_ratios]
    if step_ratios is None:
        print("请输入参数：step_ratios")
        return

    for step_ratio in step_ratios:
        # 分阶段训练函数第一阶段实验名字用step_ratio加以区分， 第二阶段实验名字在第一阶段实验名上加上_step2
        exp_name = 'score_train_2_stage_step_ratio_{}'.format(step_ratio)
        train_2_stage(exp_name, num_epochs=100, step_ratio=step_ratio)

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
        # TODO 这里ap值的横坐标应该改成epoch数目
        for key in val_info['box'].keys():
            writer.add_scalar('box_ap' + str(key), val_info['box'][key], epoch)

        for key in val_info['mask'].keys():
            writer.add_scalar('mask_ap' + str(key), val_info['mask'][key], epoch)

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()
        # TODO 这里返回测试得到的box_ap和mask_ap
        return val_info['box']['all'], val_info['mask']['all']


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


# TODO 这里需要对get_loss进行改进，添加weight, dataset,等参数，让函数更加灵活
def get_loss(weight_path):
    """
    为了获取每个样本的损失值，首先先借用train()函数的代码块，进入到计算损失的步骤
    :return: 元素为{file_name: tensor}的loss_list
    """
    loss_list = []
    with torch.no_grad():
        #  这里为dataset的构造函数添加一个参数，是否会返回file_name
        dataset = COCODetection(image_path=cfg.dataset.train_images,
                                info_file=cfg.dataset.train_info,
                                transform=SSDAugmentation(MEANS),
                                return_file_name=True)

        # Parallel wraps the underlying module, but when saving and loading we don't want that
        yolact_net = Yolact()
        net = yolact_net
        net.train()

        # I don't use the timer during training (I use a different timing method).
        # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
        timer.disable_all()
        print('Initializing weights...')
        yolact_net.load_weights(weight_path)

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
                print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (
                args.batch_alloc, args.batch_size))
                exit(-1)

        #  这里不经过 CustomDataParallel 而是 PseudoDataParallel
        net = PseudoDataParallel(NetLossSingle(net, criterion))

        if args.cuda:
            net = net.cuda()

        # Initialize everything
        if not cfg.freeze_bn: yolact_net.freeze_bn()  # Freeze bn so we don't kill our means
        yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
        if not cfg.freeze_bn: yolact_net.freeze_bn(True)

        # TODO 这里把data_loader batch_size 固定为1
        data_loader = data.DataLoader(dataset, 1,
                                      num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate,
                                      pin_memory=True)

        global loss_types  # Forms the print order

        loss_avgs = {k: MovingAverage(100) for k in loss_types}

        print('Begin computing loss!')
        name_tmp = []
        loss_tmp = []
        # try-except so you can use ctrl+c to save early and stop training
        try:
            for datum in tqdm(data_loader):
                # TODO 将datum 中的file_name拆开来
                file_name = datum[2]
                name_tmp.extend(file_name)
                datum.pop()
                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                """
                    在这里修改 不再调用NetLoss的forward方法，调用NetLossSingle的forward方法以期望获取单个样本的损失
                """
                # TODO 找机会看看格式为什么是二维数组
                losses = net(datum)
                loss_tmp.extend(losses[0])

            for name, loss in zip(name_tmp, loss_tmp):
                loss_dict = {}
                loss_dict['file_name'] = name
                loss_dict['loss'] = loss.item()
                loss_list.append(loss_dict)

        except KeyboardInterrupt:
            if args.interrupt:
                print('Stopping early.')
            exit()

        return loss_list


def get_confidence_v2(threshold=0.5):
    """
    为了获取每个样本的置信度，首先先借用train()函数的代码块，进入到计算损失的步骤
    这里修改了get_confidence之前的错误
    存储文件名根据置信度阈值和数据集名称决定
    :return: 元素为{file_name: confidence}的confidence_list
    """
    confidence_list = []
    confidence_name = 'confidence_{}_threshold_{}.pkl'.format(cfg.dataset.name, threshold)
    confidence_path = os.path.join(confidence_root, confidence_name)
    with torch.no_grad():
        #  这里为dataset的构造函数添加一个参数，是否会返回file_name
        dataset = COCODetection(image_path=cfg.dataset.train_images,
                                info_file=cfg.dataset.train_info,
                                transform=SSDAugmentation(MEANS),
                                return_file_name=True)

        # Parallel wraps the underlying module, but when saving and loading we don't want that
        yolact_net = Yolact()
        net = yolact_net
        net.train()

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

            if args.start_iter == -1:
                args.start_iter = SavePath.from_str(args.resume).iteration
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
                print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (
                args.batch_alloc, args.batch_size))
                exit(-1)

        #  这里不经过 CustomDataParallel 而是 PseudoDataParallel
        net = PseudoDataParallel(NetConfidenceSingle(net, criterion, threshold))
        # net = NetLossSingle(net, criterion)

        if args.cuda:
            net = net.cuda()

        # Initialize everything
        if not cfg.freeze_bn: yolact_net.freeze_bn()  # Freeze bn so we don't kill our means
        yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
        if not cfg.freeze_bn: yolact_net.freeze_bn(True)

        data_loader = data.DataLoader(dataset, args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate,
                                      pin_memory=True)

        global loss_types  # Forms the print order

        loss_avgs = {k: MovingAverage(100) for k in loss_types}

        print('Begin computing confidence!')
        # try-except so you can use ctrl+c to save early and stop training
        try:
            for datum in tqdm(data_loader):
                #  将datum 中的file_name拆开来
                file_name = datum[2]
                datum.pop()
                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)

                confidences = net(datum)
                confidence_tmp = []
                for confidence in confidences:
                    confidence_tmp.extend(confidence)

                for name, confidence in zip(file_name, confidence_tmp):
                    confidence_dict = {}
                    confidence_dict['file_name'] = name
                    confidence_dict['confidence'] = confidence.item()
                    confidence_list.append(confidence_dict)
                debug = 0
        except KeyboardInterrupt:
            if args.interrupt:
                print('Stopping early.')
            exit()
        with open(confidence_path, 'wb') as f:
            pickle.dump(confidence_list, f)
        return confidence_list

def get_new_data_loader():
    """
        当到了某个界值，我们应当将原始数据集替换成精炼之后的数据集
        需要我们同时改变dataset 和 data_loader
    """
    confidence_list_path = os.path.join(confidence_root, 'confidence_COCO_Person_threshold_0.3.pkl')
    with open(confidence_list_path, 'rb') as f:
        confidence_list = pickle.load(f)

    # TODO 将 confidence_list 进行排序
    confidence_list.sort(key=lambda x: x['confidence'], reverse=True)

    all_image_list = [x['file_name'] for x in confidence_list]
    threshold = args.confidence_threshold
    selected_list = all_image_list[:math.ceil(len(all_image_list) * (1))]
    dataset = build_dataset_from_list(exp_name='tmp',
                                      image_file_name_list=selected_list)
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)
    return data_loader

if __name__ == '__main__':
    """
        train_step中如果使用自己构造的数据集则会出错，不管有没有resume latest train_step(exp_name, num_epochs=stage2_num_epochs, resume='latest')
        train函数使用自己构造的数据集，且resume latest也没有问题  train(exp_name, num_epochs=stage2_num_epochs, dataset=dataset, resume='latest')
    """
    # train_2_stage("test_train_2_stage", num_epochs=4)
    train_step_by_step()