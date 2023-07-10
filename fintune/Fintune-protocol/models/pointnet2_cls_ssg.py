import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import BasicBlock

from models.pointnet2_utils import PointNetSetAbstraction




def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


class ProjEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_size = 32
        self.trans_dim = 8
        self.graph_dim = 64
        self.imgblock_dim = 64
        self.img_size = 224
        self.obj_size = 224
        self.imagenet_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.imagenet_std = torch.Tensor([0.229, 0.224, 0.225])

        self.input_trans = nn.Conv1d(3, self.trans_dim, 1)
        self.graph_layer = nn.Sequential(nn.Conv2d(self.trans_dim * 2, self.graph_dim, kernel_size=1, bias=False),
                                         nn.GroupNorm(4, self.graph_dim),
                                         nn.LeakyReLU(negative_slope=0.2)
                                         )
        self.proj_layer = nn.Conv1d(self.graph_dim, 3, kernel_size=1)

        self.img_block = self.img_block = nn.Sequential(
            BasicBlock(self.graph_dim, self.graph_dim),
            nn.Conv2d(self.graph_dim, self.graph_dim, kernel_size=1),
        )
        self.img_layer = nn.Conv2d(self.graph_dim, 3, kernel_size=1)

        self.offset = torch.Tensor([[-1, -1], [-1, 0], [-1, 1],
                                    [0, -1], [0, 0], [0, 1],
                                    [1, -1], [1, 0], [1, 1]])

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k, k):
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            idx = knn_point(k, coor_k.transpose(1, 2).contiguous(), coor_q.transpose(1, 2).contiguous())  # B G k
            idx = idx.transpose(1, 2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, original_pc, pc):
        B, N, _ = pc.shape



        # Get features
        original_pc = original_pc.transpose(1, 2).contiguous()  # B, 3, N
        # print("original_pc", original_pc.shape)
        f = self.input_trans(original_pc)
        # print("f", f.shape)
        f = self.get_graph_feature(original_pc, f, original_pc, f, self.local_size)
        # print("f", f.shape)
        f = self.graph_layer(f)
        # print("f",f.shape)
        f = self.img_block(f.contiguous())
        f = self.img_layer(f)
        f = f.max(dim=-1, keepdim=False)[0]  # B C N
        f = f.transpose(1, 2)
        # print("f", f.shape)
        # f = self.proj_layer(f).transpose(1, 2).contiguous()  # B N C
        # print("f", f.shape)

        mean_vec = self.imagenet_mean.unsqueeze(dim=0).unsqueeze(dim=1).to(f.device)  # 1 3 1
        std_vec = self.imagenet_std.unsqueeze(dim=0).unsqueeze(dim=1).to(f.device)  # 1 3 1
        # Normalize the pic
        color = nn.Sigmoid()(f)
        # print("color",color.shape)
        color_norm = color.sub(mean_vec).div(std_vec)
        out = torch.cat([pc,color_norm],dim=-1)
        # print("out",out.shape)
        return out

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
    def __init__(self,num_class,normal_channel=True,use_color=False):
        super(get_model, self).__init__()
        in_channel = 3+3 if normal_channel else 3 # 输入的坐标维度是多少就输入多少
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False) # 输入的坐标维度是多少就输入多少
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True) # 输入的坐标维度是多少就输入多少
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

        # self.pos_embedding = PosE_Initial(3,72,100,1000)

        # self.color_block = ProjEnc()
    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            # xyz = self.color_block(xyz.transpose(2, 1), xyz.transpose(2, 1))
            # print("xyz", xyz.shape)
            # xyz = xyz.transpose(2, 1)
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # xyz = self.pos_embedding(xyz)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.classifer_g2(x)
        x = F.log_softmax(x, -1)
        # print("x",x.shape)


        return x, l3_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


if __name__ == '__main__':
    a = torch.rand([4,1024,3])
    color_block = ProjEnc()
    out = color_block(a,a)
    print("out",out.shape)

