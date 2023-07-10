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
from models import pointnet2_cls_ssg as pointnet2
from data import ModelNet40Cls
import utils.pytorch_utils as pt_utils
import pointnet2_ops.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Classification Training')
parser.add_argument('--config', default='cfgs/config_ssn_cls.yaml', type=str)

def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")
    
    try:
        os.makedirs(args.save_path)
    except OSError:
        pass
    
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    # 改dataloader,预训练模型
    train_dataset = ModelNet40Cls(num_points = args.num_points, root = args.data_root, transforms=train_transforms)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=int(args.workers), 
        pin_memory=True
    )

    test_dataset = ModelNet40Cls(num_points = args.num_points, root = args.data_root, transforms=test_transforms, train=False)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers), 
        pin_memory=True
    )

    # 改模型，normal_channel存颜色
    # model = RSCNN_SSN(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    model = pointnet2.get_model(num_class = args.num_classes,normal_channel=False)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
    
    if args.checkpoint is not '':
        checkpoint = torch.load(args.checkpoint, map_location={'cuda:1': 'cuda:0'})

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
    criterion = pointnet2.get_loss()
    num_batch = len(train_dataset)/args.batch_size
    
    # training
    train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)
    

def train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    global g_acc 
    g_acc = 0.91    # only save the model whose acc > 0.91
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
            
            # fastest point sampling
            fps_idx = pointnet2_utils.furthest_point_sample(points, 1200)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            
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
    acc = (preds == labels).sum() / labels.numel()
    print('\nval loss: %0.6f \t acc: %0.6f\n' %(np.array(torch.stack(losses).cpu()).mean(), acc))
    if acc > g_acc:
        g_acc = acc
        torch.save(model.state_dict(), '%s/cls_ssn_iter_%d_acc_%0.6f.pth' % (args.save_path, iter, acc))
    model.train()
    
if __name__ == "__main__":
    main()