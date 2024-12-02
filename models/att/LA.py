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

from models.att.SimAM import SimAM

class LinAngularAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=2,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
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


    def forward(self, x):
        B, C, H, W = x.shape
        N, L, C = B, H*W, C
        x = x.permute(0,2,3,1).contiguous().view(N,L,C)
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
        dconv_v = self.dconv(v)

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
        x += dconv_v
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0,2,1).contiguous().view(N,C,H,W)
        # x = x.reshape(B, C, H, W)
        return x

class ResLA(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.LA = LinAngularAttention(dim)
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
    model = LinAngularAttention(in_channels=1024, num_heads=8, qkv_bias=False, sparse_reg=False)
    print(model)
    X = torch.ones(32, 1024, 64, 64)
    Y = model(X)
    print(Y.shape)