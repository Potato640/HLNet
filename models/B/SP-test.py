import torch
import torch.nn as nn
from einops import rearrange
from models.B.AKM import AKConv


class SimPool(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, gamma=None, use_beta=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_patches = nn.LayerNorm(dim, eps=1e-6)

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)

        if gamma is not None:
            self.gamma = torch.tensor([gamma], device='cuda')
            if use_beta:
                self.beta = nn.Parameter(torch.tensor([0.0], device='cuda'))
        self.eps = torch.tensor([1e-6], device='cuda')

        self.gamma = gamma
        self.use_beta = use_beta

        # self.convT = AKConv(dim, dim // 8, 5)
        # self.convRT = AKConv(dim // 8, dim, 5)

    def prepare_input(self, x):
        if len(x.shape) == 3:  # Transformer
            # Input tensor dimensions:
            # x: (B, N, d), where B is batch size, N are patch tokens, d is depth (channels)
            B, N, d = x.shape
            gap_cls = x.mean(-2)  # (B, N, d) -> (B, d)
            gap_cls = gap_cls.unsqueeze(1)  # (B, d) -> (B, 1, d)
            return gap_cls, x
        if len(x.shape) == 4:  # CNN
            # Input tensor dimensions:
            # x: (B, d, H, W), where B is batch size, d is depth (channels), H is height, and W is width
            print("-----CNN-----")
            B, d, H, W = x.shape
            gap_cls = x.mean([-2, -1])  # (B, d, H, W) -> (B, d)
            x = x.reshape(B, d, H * W).permute(0, 2, 1)  # (B, d, H, W) -> (B, d, H*W) -> (B, H*W, d)
            gap_cls = gap_cls.unsqueeze(1)  # (B, d) -> (B, 1, d)
            return gap_cls, x
        else:
            raise ValueError(f"Unsupported number of dimensions in input tensor: {len(x.shape)}")

    def forward(self, x):
        B, C, H, W = x.shape
        tmp = x

        # Prepare input tensor and perform GAP as initialization
        gap_cls, x = self.prepare_input(x)
        print("gap_cls--",gap_cls.shape)

        # Prepare queries (q), keys (k), and values (v)
        q, k, v = gap_cls, self.norm_patches(x), self.norm_patches(x)

        # Extract dimensions after normalization
        Bq, Nq, dq = q.shape
        print("q--",q.shape)
        Bk, Nk, dk = k.shape
        print("k--", k.shape)
        Bv, Nv, dv = v.shape
        print("v--", v.shape)

        # Check dimension consistency across batches and channels
        assert Bq == Bk == Bv
        assert dq == dk == dv

        # Apply linear transformation for queries and keys then reshape
        qq = self.wq(q).reshape(Bq, Nq, self.num_heads, dq // self.num_heads).permute(0, 2, 1,
                                                                                      3)  # (Bq, Nq, dq) -> (B, num_heads, Nq, dq/num_heads)

        print("qq--",qq.shape)
        kk = self.wk(k).reshape(Bk, Nk, self.num_heads, dk // self.num_heads).permute(0, 2, 1,
                                                                                      3)  # (Bk, Nk, dk) -> (B, num_heads, Nk, dk/num_heads)
        print("kk--", kk.shape)
        vv = v.reshape(Bv, Nv, self.num_heads, dv // self.num_heads).permute(0, 2, 1,
                                                                             3)  # (Bv, Nv, dv) -> (B, num_heads, Nv, dv/num_heads)
        print("vv--", vv.shape)
        # Compute attention scores
        attn = (qq @ kk.transpose(-2, -1)) * self.scale
        print("attn--", attn.shape)
        # Apply softmax for normalization
        attn = attn.softmax(dim=-1)
        print("soft-attn--", attn.shape)

        # If gamma scaling is used
        if self.gamma is not None:
            # Apply gamma scaling on values and compute the weighted sum using attention scores
            x = torch.pow(attn @ torch.pow((vv - vv.min() + self.eps), self.gamma),
                          1 / self.gamma)  # (B, num_heads, Nv, dv/num_heads) -> (B, 1, 1, d)
            # If use_beta, add a learnable translation

            if self.use_beta:
                x = x + self.beta
        else:
            # Compute the weighted sum using attention scores
            x = (attn @ vv).transpose(1, 2).reshape(Bq, dq, Nq, Nq)
            print("x--", (attn @ vv).transpose(1, 2).shape)
            print("x--",x.shape)

            x = x * tmp

            # x = x + tmp

            # x = tmp2 + x


        return x


if __name__ == '__main__':
    model = SimPool(32)
    print(model)
    X = torch.ones(32, 32, 64, 64)
    Y = model(X)
    print(Y.shape)