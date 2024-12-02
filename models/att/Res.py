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

