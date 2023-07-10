import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils_msg_vote import PointNetSetAbstractionMsg, PointNetSetAbstraction

class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        B, _, N = xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        # feat_dim = self.out_dim
        # print("feat_dim",feat_dim)
        feat_range = torch.arange(feat_dim).float().cuda()
        # print("feat_range", feat_range)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim) #以self.alpha为底的feat_range / feat_dim次方
        # print("dim_embed", dim_embed)
        # print("xyz.unsqueeze(-1)",xyz.unsqueeze(-1).shape)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed) #self.beta * xyz.unsqueeze(-1)除以dim_embed
        # print("div_embed", div_embed.shape)
        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        # print("torch.stack([sin_embed, cos_embed], dim=4)",torch.stack([sin_embed, cos_embed], dim=4).shape)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)

        return position_embed
class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
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

        # self.pos_embedding = PosE_Initial(2,72,1000,100)
        # self.proj = nn.Sequential(
        #     nn.Linear(72, 3),
        #     nn.ReLU(),
        #     nn.Linear(3, 3),
        #
        # )
    def forward(self, xyz):
        B, _, _ = xyz.shape

        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # xyz = self.pos_embedding(xyz)
        # # print("xyz",xyz.shape)
        # xyz = self.proj(xyz.transpose(2,1))
        # # print("xyz", xyz.shape)
        # xyz = xyz.transpose(2,1)

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.classifer_g2(x)
        # x = F.log_softmax(x, -1)


        return x,l3_points


#class get_loss(nn.Module):
#    def __init__(self):
#        super(get_loss, self).__init__()
#
#    def forward(self, pred, target, trans_feat):
#        total_loss = F.nll_loss(pred, target)
#
#        return total_loss

class get_loss(nn.Module):

    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat,smoothing=True):

        target = target.contiguous().view(-1)
        if smoothing:
            eps = 0.2
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, target, reduction='mean')

        return loss


