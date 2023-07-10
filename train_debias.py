"""
Author: Benny
Date: Nov 2019
"""
import os

import os
import sys
import torch
import numpy as np

import datetime
import logging

import torchvision
from torch import einsum, mean
from torch.cuda.amp import autocast
from torch.nn import init
from torchvision.transforms import transforms

# import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.Pixels_loader import PixelsLoader
from tensorboardX import SummaryWriter
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
from apex import amp
from torch.utils.data import DistributedSampler


import torch.distributed as dist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))



def Reg(g,b):
    reg_loss = 0.0
    for i in range(len(g)):
        # print("i",i)
        # print("g[i]",g[i])
        g_in = torch.nn.functional.normalize(g[i], p=2, dim=-1)
        b_in = torch.nn.functional.normalize(b[i], p=2, dim=-1)
        # print("g_in",g_in.shape)
        # print("b_in",b_in.shape)
        # print("len(g_in.shape)",len(g_in.shape))
        if len(g_in.shape)==2:
            # print("len(g_in.shape)", len(g_in.shape))

            reg = einsum("kmd,kd->km", b_in, g_in)
            reg_loss = reg_loss+ torch.mean(torch.sum(reg ** 2, dim=-1))

        # if args.local_rank ==1:
        #     print("reg_loss",reg_loss)
    return reg_loss

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=40, help='batch size in training')
    parser.add_argument('--model', default='pointnet_img', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=55, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


def test(model, loader, num_class=13,val_writer=None,epoch=0,criterion=None,test_sample=None):
    # device = torch.device("cuda:1" if args.cuda else "cpu")
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    loss = 0.0
    test_sample.set_epoch(epoch)
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.type(torch.float32).cuda(args.local_rank), target.cuda(args.local_rank)
        # pos = (points[:, :, :2]).to(torch.float32)
        # rgb = (points[:, :, 2:]).to(torch.float32)
        # pos = encoding(pos)
        # points = torch.cat([pos, rgb], dim=-1)
        points = points.transpose(2, 1)
        # points = points[:,:2,:]
        # with autocast():
        pred, trans_feat,_,_,_ = classifier(points)
        # loss = criterion(pred, target.long(), trans_feat)
        pred_choice = pred.data.max(1)[1] # 格式：（B）
        # print("pred_choice",pred_choice.shape)
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    # print("class_acc[:, 2]",class_acc[:, 2])
    # print("class_acc[:, 0]", class_acc[:, 0])
    # print("class_acc[:, 2]", class_acc[:, 1])
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    if val_writer is not None:
        val_writer.add_scalar('Metric/loss', loss, epoch)
        val_writer.add_scalar('Metric/instance_acc', instance_acc, epoch)
        val_writer.add_scalar('Metric/class_acc', class_acc, epoch)
    return instance_acc, class_acc


def main(args):
    torch.distributed.init_process_group("nccl", world_size=2, rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    def log_string(str):
        logger.info(str)
        print(str)


    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    tf_dir = exp_dir.joinpath('tfboard/')
    tf_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    """tensorboard"""
    train_writer = None
    val_writer = None
    if args.local_rank == 0:
        train_writer = SummaryWriter(os.path.join(tf_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(tf_dir, 'test'))



    '''DATA LOADING'''
    log_string('Load dataset ...')

    datasets = PixelsLoader()
    train_size = int(len(datasets) * 0.7)
    # print("train_size",train_size)
    test_size = len(datasets) - train_size
    # test_size = int(len(datasets) * 0.04)
    train_dataset, test_dataset = torch.utils.data.random_split(datasets, [train_size, test_size])
    train_sample = DistributedSampler(train_dataset)
    test_sample = DistributedSampler(test_dataset)
    # print("ok")
    trainDataLoader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=22,
        sampler=train_sample,
    )
    testDataLoader = torch.utils.data.DataLoader( dataset=test_dataset, batch_size=args.batch_size, num_workers=22,
                                                  drop_last=True,sampler=test_sample)
    # print("ok")
    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = convert_syncbn_model(model.get_model(num_class, normal_channel=args.use_normals)).cuda(args.local_rank)
    criterion = model.get_loss().cuda(args.local_rank)
    classifier.apply(inplace_relu)
    # print(classifier)
    # if not args.use_cpu:
    #     classifier = classifier.cuda()
    #     criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0



    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    classifier, optimizer = amp.initialize(classifier, optimizer, opt_level="O1")
    classifier = DistributedDataParallel(classifier, delay_allreduce=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)


    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()

        n_batches = len(trainDataLoader)
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            n_itr = epoch * n_batches + batch_id
            if not args.use_cpu:
                points, target = points.type(torch.float32).cuda(args.local_rank), target.cuda(args.local_rank)
            points = points.transpose(2, 1)
            B,_,N = points.shape
            zeros = torch.zeros(B,1,N).cuda(args.local_rank)
            # with autocast():
            pred, trans_feat,pre_bias,g,b = classifier(points,zeros)
            loss_ge = criterion(pred, target.long(), trans_feat)
            loss_bi = criterion(pre_bias, target.long(), trans_feat)
            l_reg = Reg(g,b)
            # print("loss_ge",loss_ge)
            # print("loss_bi", loss_bi)
            # print("l_reg", l_reg)
            if epoch < 6:
                loss = loss_bi + l_reg
            else:
                loss = loss_ge + loss_bi + l_reg
            # print("pred",pred.shape)
            # print("target.long()", len(target.long()))
            pred_choice = pred.data.max(1)[1]
            # print("pred_choice",pred_choice.shape)
            # print("target.long().data", target.long().data)
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            global_step += 1

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class,val_writer=val_writer,epoch=epoch,criterion=criterion,test_sample=test_sample)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
