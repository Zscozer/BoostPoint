import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sklearn.metrics as metrics
import os
from torchvision import transforms
from models import RSCNN_SSN_Cls as RSCNN_SSN
from data import ModelNet40Cls
import utils.pytorch_utils as pt_utils
from data import ScanObjectNN
from data import ScanObjectNN_hardest
from data import  ScanObjectNN_color
import pointnet2_ops.pointnet2_utils as pointnet2_utils
from models import pointnet2_cls_ssg
from models import pointnet2_img_cls_msg_vote as pointnet2_img_cls_msg
import data.data_utils as d_utils
import argparse
import random
import yaml

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Classification Voting Evaluation')
parser.add_argument('--config', default='cfgs/config_pointnet2_vote_cls_objNN.yaml', type=str)
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


NUM_REPEAT = 300
NUM_VOTE = 10

def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    for k, v in config['common'].items():
        setattr(args, k, v)

    seed = args.seed
    # seed = random.randint(120,200)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.use_hard_model:
        test_dataset = ScanObjectNN_hardest(args,num_points=args.num_points, root=args.data_root, train=False)
    else:
        if args.use_color:
            test_dataset = ScanObjectNN_color(num_points=args.num_points, partition='test')
        else:
            test_dataset = ScanObjectNN(args,num_points=args.num_points, root=args.data_root, train=False)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True
    )

   # model = RSCNN_SSN(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)

    if args.use_pointnet2_msg:
        if args.use_color:
            model = pointnet2_img_cls_msg.get_model(num_class = 15,normal_channel=True)
        else:
            model = pointnet2_img_cls_msg.get_model(num_class = 40,normal_channel=False)
    else:
        model = pointnet2_cls_ssg.get_model(num_class=args.num_classes, normal_channel=False)
    model.cuda()
    
    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))
    
    # evaluate
    PointcloudScale = d_utils.PointcloudScale()   # initialize random scaling
    model.eval()
    global_acc = 0
    global_mean_acc = 0
    for i in range(NUM_REPEAT):
        preds = []
        labels = []

        for j, data in enumerate(test_dataloader, 0):
            points, target = data
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points, volatile=True), Variable(target, volatile=True)
            
            # fastest point sampling
            fps_idx = pointnet2_utils.furthest_point_sample(points, 1200)  # (B, npoint)
            pred = 0
            for v in range(NUM_VOTE):
                new_fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
                new_points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), new_fps_idx).transpose(1, 2).contiguous()
                if v > 0:
                    new_points.data = PointcloudScale(new_points.data)
                # pred += F.softmax(model(new_points), dim = 1)
                pred += F.softmax(model(new_points.transpose(2,1))[0], dim = 1)
            pred /= NUM_VOTE
            target = target.view(-1)
            _, pred_choice = torch.max(pred.data, -1)
            
            preds.append(pred_choice)
            labels.append(target.data)
    
        preds = torch.cat(preds, 0)
        labels = torch.cat(labels, 0)

        preds = preds.cpu()
        labels = labels.cpu()
        acc = 100. * metrics.accuracy_score(labels, preds)
        mean_acc = 100. * metrics.balanced_accuracy_score(labels, preds)

        if acc > global_acc:
            global_acc = acc
        if mean_acc > global_mean_acc:
            global_mean_acc = mean_acc
        print('Repeat %3d \t mean Acc: %0.6f\t Acc: %0.6f' % (i + 1, mean_acc , acc))

    print('\nBest voting mean acc: %0.6f \tBest voting acc: %0.6f ' % (global_mean_acc, global_acc))
        
if __name__ == '__main__':
    main()

