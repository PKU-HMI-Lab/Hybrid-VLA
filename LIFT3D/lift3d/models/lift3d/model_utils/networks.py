# Parametric Networks for 3D Point Cloud Classification
import pytorch3d.ops as torch3d_ops
import torch
import torch.nn as nn

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
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
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


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
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        _, fps_idx = torch3d_ops.sample_farthest_points(points=xyz, K=self.group_num)
        fps_idx = torch.randint(0, N, (B, self.group_num)).long()
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)
        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)
        # lc_xyz&lc_x is cernter point&feature, knn_xyz&knn_x is neighbor point&feature
        return lc_xyz, lc_x, knn_xyz, knn_x, fps_idx, knn_idx


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(
        self, out_dim, alpha, beta, block_num, dim_expansion, type, adapter_layer=0
    ):
        super().__init__()
        self.type = type
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)
        if dim_expansion == 1:
            expand = 2
        elif dim_expansion == 2:
            expand = 1

        self.adapter_layer = adapter_layer
        self.linear2 = []
        for i in range(block_num):
            self.linear2.append(
                Linear2Layer(out_dim, bias=True, adapter_layer=adapter_layer)
            )
        self.linear2 = nn.Sequential(*self.linear2)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):

        # Normalization
        if self.type == "mn40":
            mean_xyz = lc_xyz.unsqueeze(dim=-2)
            std_xyz = torch.std(knn_xyz - mean_xyz)
            knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        elif self.type == "scan":
            knn_xyz = knn_xyz.permute(0, 3, 1, 2)
            knn_xyz -= lc_xyz.permute(0, 2, 1).unsqueeze(-1)
            knn_xyz /= torch.abs(knn_xyz).max(dim=-1, keepdim=True)[0]
            knn_xyz = knn_xyz.permute(0, 2, 3, 1)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        # concate the knn_x with the max of dim 2

        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)
        ##take neighbor to do
        # Linear
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x = knn_x.reshape(B, -1, G, K)

        # Geometry Extraction
        knn_x_w = self.geo_extract(knn_xyz, knn_x)
        # Linear
        for layer in self.linear2:
            knn_x_w = layer(knn_x_w)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0]
        return lc_x


# Linear layer 1
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, x):
        return self.net(x)


# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(
        self, in_channels, kernel_size=1, groups=1, bias=True, adapter_layer=0
    ):
        super(Linear2Layer, self).__init__()
        basic_dim = 32
        self.act = nn.ReLU(inplace=True)
        if adapter_layer == 2:
            self.net1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=int(32),
                    kernel_size=kernel_size,
                    groups=groups,
                    bias=bias,
                ),
                nn.BatchNorm2d(int(32)),
                self.act,
            )
            self.net2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=int(32),
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    bias=bias,
                ),
                nn.BatchNorm2d(in_channels),
            )
        else:
            self.net1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=int(in_channels / 2),
                    kernel_size=kernel_size,
                    groups=groups,
                    bias=bias,
                ),
                nn.BatchNorm2d(int(in_channels / 2)),
                self.act,
            )
            self.net2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=int(in_channels / 2),
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    bias=bias,
                ),
                nn.BatchNorm2d(in_channels),
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape  
        feat_dim = self.out_dim // (self.in_dim * 2)  

        feat_range = torch.arange(feat_dim).float().to(knn_xyz.device)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)  
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)  
        cos_embed = torch.cos(div_embed)
        position_embed = torch.cat([sin_embed, cos_embed], -1)
        position_embed = position_embed.permute(
            0, 1, 4, 2, 3
        ).contiguous()  
        position_embed = position_embed.view(B, self.out_dim, G, K)
        knn_x_w = knn_x + position_embed

        return knn_x_w


class EncP(nn.Module):
    def __init__(
        self,
        in_channels,
        input_points,
        num_stages,
        embed_dim,
        k_neighbors,
        alpha,
        beta,
        LGA_block,
        dim_expansion,
        type,
    ):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = Linear1Layer(in_channels, self.embed_dim, bias=False)

        self.FPS_kNN_list = nn.ModuleList() 
        self.LGA_list = nn.ModuleList()  
        self.Pooling_list = nn.ModuleList() 

        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * dim_expansion[i]
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(
                LGA(
                    out_dim,
                    self.alpha,
                    self.beta,
                    LGA_block[i],
                    dim_expansion[i],
                    type,
                    adapter_layer=i,
                )
            )

            self.Pooling_list.append(Pooling(out_dim))

    def forward(self, xyz, x):

        # Raw-point Embedding
        x = self.raw_point_embed(x) 
        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x, _, _ = self.FPS_kNN_list[i](
                xyz, x.permute(0, 2, 1)
            )
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)

        return xyz, x


class Point_PN_scan(nn.Module):
    def __init__(
        self,
        in_channels=3,
        class_num=15,
        input_points=1024,
        num_stages=3,
        embed_dim=96,
        beta=100,
        alpha=1000,
        LGA_block=[2, 1, 1, 1],
        dim_expansion=[2, 2, 2, 1],
        type="mn40",
        k_neighbors=81,
    ):
        super().__init__()
        # Parametric Encoder
        self.EncP = EncP(
            in_channels,
            input_points,
            num_stages,
            embed_dim,
            k_neighbors,
            alpha,
            beta,
            LGA_block,
            dim_expansion,
            type,
        )
        self.out_channels = embed_dim
        for i in dim_expansion:
            self.out_channels *= i

    def forward(self, x, xyz):
        xyz, x = self.EncP(xyz, x)
        return xyz, x
