import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

def weight_stack(b1,b2,b3,b4,b5):
    b = torch.stack([b1,b2,b3,b4,b5],dim=0)

    if len(b.shape) ==3:
        # print("b", b.shape)
        b = b.permute(1,0,2) # K M D
    # else:
    #     b = b.transpose(0,1) # K M
    return b
class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 2, [256, 512, 1024], True)
        self.classifer_g2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_class)
        )
        self.classifer_b1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_class)
        )
        self.classifer_b2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_class)
        )
        self.classifer_b3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_class)
        )
        self.classifer_b4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_class)
        )
        self.classifer_b5 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_class)
        )

    def forward(self, xyz):
        B, _, _ = xyz.shape

        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None


        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x_g2 = self.classifer_g2(x)
        x_g2 = F.log_softmax(x_g2, -1)

        x_b1 = self.classifer_b1(x)
        x_b1 = F.log_softmax(x_b1, -1)

        x_b2 = self.classifer_b2(x)
        x_b2 = F.log_softmax(x_b2, -1)

        x_b3 = self.classifer_b3(x)
        x_b3 = F.log_softmax(x_b3, -1)

        x_b4 = self.classifer_b4(x)
        x_b4 = F.log_softmax(x_b4, -1)

        x_b5 = self.classifer_b5(x)
        x_b5 = F.log_softmax(x_b5, -1)

        x_b = torch.stack([x_b1, x_b2, x_b3, x_b4, x_b5], dim=0)
        x_b = x_b.permute(1, 2, 0)

        x_b = torch.logsumexp(x_b, -1)
        # print("x_b", x_b.shape)

        g_0 = self.classifer_g2[0].weight
        # g_1 = self.classifer_g2[1].weight
        g_4 = self.classifer_g2[4].weight
        # g_5 = self.classifer_g2[5].weight
        g_8 = self.classifer_g2[8].weight
        # print("g_4", g_4.shape)
        b_0 = weight_stack(self.classifer_b1[0].weight, self.classifer_b2[0].weight, self.classifer_b3[0].weight,
                           self.classifer_b4[0].weight, self.classifer_b5[0].weight)
        # b_1 = weight_stack(self.classifer_b1[1].weight, self.classifer_b2[1].weight, self.classifer_b3[1].weight,
        #                    self.classifer_b4[1].weight, self.classifer_b5[1].weight)
        b_4 = weight_stack(self.classifer_b1[4].weight, self.classifer_b2[4].weight, self.classifer_b3[4].weight,
                           self.classifer_b4[4].weight, self.classifer_b5[4].weight)
        # b_5 = weight_stack(self.classifer_b1[5].weight, self.classifer_b2[5].weight, self.classifer_b3[5].weight,
        #                    self.classifer_b4[5].weight, self.classifer_b5[5].weight)
        b_8 = weight_stack(self.classifer_b1[8].weight, self.classifer_b2[8].weight, self.classifer_b3[8].weight,
                           self.classifer_b4[8].weight, self.classifer_b5[8].weight)
        # print("b_4",b_4.shape)
        # b = torch.stack([b1,b2,b3,b4,b5],dim=0).permute(0,)

        # print("x",x.shape)

        return x_g2, l3_points, x_b, (g_0, g_4, g_8), (b_0, b_4, b_8)


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


