import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import os
from torchvision import transforms
#from models import RSCNN_SSN_Cls as RSCNN_SSN
from models import pointnet2_cls_ssg
from models import pointnet2_img_cls_msg
from data import ModelNet40Cls
from data import ScanObjectNN
from data import ScanObjectNN_hardest
from data import ScanObjectNN_color
import utils.pytorch_utils as pt_utils
import pointnet2_ops.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True




parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Classification Training')
parser.add_argument('--config', default='cfgs/config_ssn_cls.yaml', type=str)
parser.add_argument('--seed', type=int, default=126)

# argument for pointWOLF
parser.add_argument('--PointWOLF',action='store_true',help='UsePointWOLF')
parser.add_argument('--w_num_anchor',type=int,default=4,help='Numofanchorpoint')
parser.add_argument('--w_sample_type',type=str,default='fps',help='Samplingmethodforanchorpoint,option:(fps,random)')
parser.add_argument('--w_sigma',type=float,default=0.5,help='Kernelbandwidth')
parser.add_argument('--w_R_range',type=float,default=10,help='Maximumrotationrangeoflocaltransformation')
parser.add_argument('--w_S_range',type=float,default=3,help='Maximumscailingrangeoflocaltransformation')
parser.add_argument('--w_T_range',type=float,default=0.25, help='Maximumtranslationrangeoflocaltransformation')
parser.add_argument('--AugTune',action='store_true',help='UseAugTune')


def record_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")

    return wrapper


@record_time
def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")

    args.seed = random.randint(50, 200)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("seed=", seed)

    try:
        os.makedirs(args.save_path)
    except OSError:
        pass

    if args.use_hard_model:
        train_dataset = ScanObjectNN_hardest(args,num_points=args.num_points, root=args.data_root)
        test_dataset = ScanObjectNN_hardest(args,num_points=args.num_points, root=args.data_root, train=False)
    else:
        if args.use_color:
            train_dataset = ScanObjectNN_color(num_points=args.num_points)
            test_dataset = ScanObjectNN_color(num_points=args.num_points, partition='test')
        else:
            train_dataset = ScanObjectNN(args,num_points=args.num_points, root=args.data_root)
            test_dataset = ScanObjectNN(args,num_points=args.num_points, root=args.data_root, train=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True
    )


    # 改模型，normal_channel存颜色
    # model = RSCNN_SSN(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    if args.use_pointnet2_msg:
        model = pointnet2_img_cls_msg.get_model(num_class = 55,normal_channel=True)
    else:
        # ssg
        model = pointnet2_cls_ssg.get_model(num_class = args.num_classes,normal_channel=True)
    model.cuda()

    # Adam优化
    #
    # optimizer = optim.Adam( model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    # lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    # bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    # lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    # bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)

    # SGD优化
    optimizer = optim.SGD(model.parameters(),lr=args.SGD_lr,momentum=args.SGD_momentum,weight_decay=1e-4)
    lr_scheduler = lr_sched.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-4)
    bnm_scheduler = None


    if args.checkpoint is not '':
        checkpoint = torch.load(args.checkpoint, map_location={'cuda:4': 'cuda:0'})
        # 共有三种模型：with color model / new model / old model
        if args.use_pointnet2_msg:
            if args.use_color:
                # with color model
                model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()},
                                  strict=False)
                model.classifer_g2[8] = nn.Linear(256, 15).cuda()
            else:
                # 新预训练模型的输入通道是3，旧预训练模型的输入通道是5，需要更改pointnet2_img_cls_msg.py与point2_utils.py各一处,并解除所有的注释部分
                # 新模型为“+ 3”，旧模型为“+ 2”，因为新的预训练模型在训练的时候使用3维读入图片（x,y,0）,这样加载预训练模型的时候删去的参数较少。

                # model.load_state_dict(checkpoint['model_state_dict'])

                # del checkpoint['model_state_dict']['module.classifer_g2.8.weight']
                # del checkpoint['model_state_dict']['module.classifer_g2.8.bias']

                model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()},
                                  strict=False)
                model.sa1.conv_blocks[0][0] = nn.Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1)).cuda()
                model.sa1.conv_blocks[1][0] = nn.Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1)).cuda()
                model.sa1.conv_blocks[2][0] = nn.Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1)).cuda()
                # model.sa2.conv_blocks[0][0] = nn.Conv2d(323, 64, kernel_size=(1, 1), stride=(1, 1)).cuda()
                # model.sa2.conv_blocks[1][0] = nn.Conv2d(323, 128, kernel_size=(1, 1), stride=(1, 1)).cuda()
                # model.sa2.conv_blocks[2][0] = nn.Conv2d(323, 128, kernel_size=(1, 1), stride=(1, 1)).cuda()
                # model.sa3.mlp_convs[0] = nn.Conv2d(643, 256, kernel_size=(1, 1), stride=(1, 1)).cuda()
                model.classifer_g2[8] = nn.Linear(256, 40).cuda()
            print("use pretrain model")
        else:
            # ssg

            del checkpoint['model_state_dict']['module.sa1.mlp_convs.0.weight']
            del checkpoint['model_state_dict']['module.sa2.mlp_convs.0.weight']
            del checkpoint['model_state_dict']['module.sa3.mlp_convs.0.weight']
            del checkpoint['model_state_dict']['module.classifer_g2.8.weight']
            del checkpoint['model_state_dict']['module.classifer_g2.8.bias']
            # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()},
                              strict=False)
        print('Load model successfully: %s' % (args.checkpoint))

    # 改loss
    # criterion = nn.CrossEntropyLoss()
    if args.use_pointnet2_msg:
        criterion = pointnet2_img_cls_msg.get_loss()
    else:
        criterion = pointnet2_cls_ssg.get_loss()

    num_batch = len(train_dataset)/args.batch_size
    
    # training
    train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)


def train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    global g_acc
    if args.use_hard_model:
        g_acc = 0.813
    else:
        # only save the model whose acc > 0.87
        # g_acc = 0.863
        g_acc = 0.87

    batch_count = 0
    model.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch-1)
            points, target = data
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points), Variable(target)
            
            # # fastest point sampling
            # fps_idx = pointnet2_utils.furthest_point_sample(points, 1200)  # (B, npoint)
            # fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
            # points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)


            # augmentation
            points.data = PointcloudScaleAndTranslate(points.data)
            
            optimizer.zero_grad()
            
            pred ,trans_feat= model(points.transpose(2,1))
            target = target.view(-1)

            loss = criterion(pred, target.long(), trans_feat)
            loss.backward()
            optimizer.step()
            if i % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' %(epoch+1, i, num_batch, loss.data.clone(), lr_scheduler.get_lr()[0]))
            batch_count += 1
            
            # validation in between an epoch
            if args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
                validate(test_dataloader, model, criterion, args, batch_count)


def validate(test_dataloader, model, criterion, args, iter): 
    global g_acc
    model.eval()
    losses, preds, labels = [], [], []
    for j, data in enumerate(test_dataloader, 0):
        points, target = data
        points, target = points.cuda(), target.cuda()
        points, target = Variable(points, volatile=True), Variable(target, volatile=True)
        
        # fastest point sampling
        fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
        # fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

        pred,trans_feat = model(points.transpose(2,1))
        target = target.view(-1)
        loss = criterion(pred, target.long(),trans_feat)
        losses.append(loss.data.clone())
        _, pred_choice = torch.max(pred.data, -1)
        
        preds.append(pred_choice)
        labels.append(target.data)
        
    preds = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)
    # OA精度
    acc = (preds == labels).sum() / labels.numel()
    print('\nval loss: %0.6f \t acc: %0.6f\n' %(np.array(torch.stack(losses).cpu()).mean(), acc))
    if acc > g_acc:
        g_acc = acc
        torch.save(model.state_dict(), '%s/cls_msg_iter_%d_acc_%0.6f_seed_%d_with_color_model_2048.pth' % (args.save_path, iter, acc, args.seed ))
    model.train()
    
if __name__ == "__main__":
    main()