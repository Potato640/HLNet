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

        # Prepare queries (q), keys (k), and values (v)
        q, k, v = gap_cls, self.norm_patches(x), self.norm_patches(x)

        # Extract dimensions after normalization
        Bq, Nq, dq = q.shape
        Bk, Nk, dk = k.shape
        Bv, Nv, dv = v.shape

        # Check dimension consistency across batches and channels
        assert Bq == Bk == Bv
        assert dq == dk == dv

        # Apply linear transformation for queries and keys then reshape
        qq = self.wq(q).reshape(Bq, Nq, self.num_heads, dq // self.num_heads).permute(0, 2, 1,
                                                                                      3)  # (Bq, Nq, dq) -> (B, num_heads, Nq, dq/num_heads)
        kk = self.wk(k).reshape(Bk, Nk, self.num_heads, dk // self.num_heads).permute(0, 2, 1,
                                                                                      3)  # (Bk, Nk, dk) -> (B, num_heads, Nk, dk/num_heads)

        vv = v.reshape(Bv, Nv, self.num_heads, dv // self.num_heads).permute(0, 2, 1,
                                                                             3)  # (Bv, Nv, dv) -> (B, num_heads, Nv, dv/num_heads)

        # Compute attention scores
        attn = (qq @ kk.transpose(-2, -1)) * self.scale

        # Apply softmax for normalization
        attn = attn.softmax(dim=-1)


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

            x = x * tmp



        return x
        cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


class SPAK(nn.Module):
    def __init__(self, dim, dim2):
        super().__init__()
        self.sm = SimPool(dim * 2)
        self.ak = AKConv(dim // 2 , dim // 2, 4)
        self.conv1 = nn.Conv2d(dim,2 * dim,1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)

        self.conv11 = nn.Conv2d(2 * dim, dim, 1)
        self.conv22 = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        a = self.conv1(x)
        x1 = self.sm(a)

        b = self.conv2(x)
        x2 = self.ak(b)

        x11 = self.conv11(x1)
        x22 = self.conv22(x2)

        x = x11 + x22
        return x



if __name__ == '__main__':
    model = SPAK(32,32)
    print(model)
    X = torch.ones(32, 32, 64, 64)
    Y = model(X)
    print(Y.shape)
