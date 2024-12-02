import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# if use Q @ K, FLOPs caclulation could be wrong
class MatMul(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        out = a @ b
        return out

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=2,
        res_kernel_size=9,
            mlp_ratio=4.,
            act_layer=nn.SiLU,
            drop=0.,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        assert in_channels % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads


        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
                                  nn.BatchNorm2d(in_channels // 4))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels // 4, 5, stride=1, padding=6, dilation=3))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels, kernel_size=1))

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )
        self.norm2 = norm_layer(in_channels)
        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(in_features=in_channels, hidden_features=mlp_hidden_dim,out_features=in_channels, act_layer=act_layer, drop=drop)


    def forward(self, x):
        res = x
        n1 = self.conv(x)

        n2 = self.conv2(n1)

        n3 = self.conv3(n2)
        # print("nq.shape", n3.shape)

        n3 = n3 + res


        B, C, H, W = x.shape
        # print(x.shape)
        N, L, C = B, H * W, C
        n3 = n3.permute(0, 2, 3, 1).contiguous().view(N, L, C)
        n3 = torch.nn.functional.relu(n3)

        x = x.permute(0,2,3,1).contiguous().view(N,L,C)

        # mid = x.reshape(N,L,self.num_heads,C // self.num_heads).permute(0,2,1,3)
        # print(mid.shape)

        MLPIO = self.mlp(self.norm2(x))
        # # print(MLPIO.shape)
        # qkv = (
        #     self.qkv(x)
        #     .reshape(N, L, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        # )
        # q, k, v= qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        #
        # q = q / q.norm(dim=-1, keepdim=True)
        # k = k / k.norm(dim=-1, keepdim=True)
        # # print("q",q.shape)
        # # print("k", k.shape)
        #
        # attn = self.kq_matmul(k.transpose(-2, -1), q)
        # # attn = attn / attn.norm(dim=-1, keepdim=True)
        # # print(attn.shape)
        # # attn = attn.reshape(N,L,C)
        # # print("qk",attn.shape)
        #
        # x = self.kq_matmul(mid, attn)
        # x = x / x.norm(dim=-1, keepdim=True)

        # x = x.permute(0, 1, 3, 2).contiguous().view(N, L, C)
        x = n3 + MLPIO

        x = x / x.norm(dim=-1, keepdim=True)

        x = x.permute(0,2,1).contiguous().view(N,C,H,W)

        return x

class ResCN(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.cn = CN(dim)
        self.conv = nn.Conv2d(dim,dim,1)

    def forward(self, x):
        res1 = self.conv(x)
        res2 = self.conv(x)
        res3 = self.conv(res2)
        x = self.cn(res3)
        res4 = self.conv(x)
        x2 = res2 + res4
        x3 = x2 + res1
        x4 = self.conv(x3)
        return  x4

if __name__ == '__main__':
    model = CN(in_channels=1024, num_heads=8)
    # print(model)
    X = torch.ones(32, 1024, 64, 64)
    Y = model(X)
    # print(Y.shape)