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
        # print(x.shape)
        return x

class SSS(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
            mlp_ratio=4.,
            act_layer=nn.SiLU,
            drop=0.,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        assert in_channels % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kq_matmul = MatMul()
        self.kqv_matmul = MatMul()
        if self.sparse_reg:
            self.qk_matmul = MatMul()
            self.sv_matmul = MatMul()

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
        B, C, H, W = x.shape
        N, L, C = B, H * W, C
        x = x.permute(0, 2, 3, 1).contiguous().view(N, L, C)
        # print(x.shape)
        mid = self.mlp(self.norm2(x))
        mid = mid.reshape(N, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        qkv = (
            self.qkv(x)
            .reshape(N, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.sparse_reg:
            out = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
            # attn = attn.softmax(dim=-1)
            # mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            # sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        # dconv_v = self.dconv(v)
        # print("dconv_v.shape", dconv_v.shape)

        attn = self.kq_matmul(k.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                    self.sv_matmul(out, v)
                    + 0.5 * v
                    + 1.0 / math.pi * self.kqv_matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * self.kqv_matmul(q, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x += mid
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 1).contiguous().view(N, C, H, W)
        # x = x.reshape(B, C, H, W)
        return x

class ResST(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.LA = SSS(dim)
        self.conv = nn.Conv2d(dim,dim,1)

    def forward(self, x):
        res1 = self.conv(x)
        res2 = self.conv(x)
        res3 = self.conv(res2)
        x = self.LA(res3)
        res4 = self.conv(x)
        x2 = res2 + res4
        x3 = x2 + res1
        x4 = self.conv(x3)
        return  x4



if __name__ == '__main__':
    model = SSS(1024)
    print(model)
    X = torch.ones(32, 1024, 64, 64)
    Y = model(X)
    print(Y.shape)