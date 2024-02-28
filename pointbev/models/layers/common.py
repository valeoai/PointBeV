import spconv.pytorch as spconv
from torch import nn

algo = spconv.ConvAlgo.Native


class MLP(nn.Module):
    def __init__(
        self,
        in_c,
        mid_c,
        out_c,
        num_layers=1,
        act=nn.ReLU(inplace=True),
        skip=True,
        as_conv=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [mid_c] * (num_layers - 1)
        self.act = act
        self.layers = nn.ModuleList(
            nn.Linear(n, k) if not as_conv else nn.Conv2d(n, k, 1)
            for n, k in zip([in_c] + h, h + [out_c])
        )
        self.skip = (in_c == mid_c) and skip
        self.as_conv = as_conv

    def forward(self, x):
        if self.as_conv:
            assert x.dim() >= 4
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = self.act(layer(x)) + x if self.skip else self.act(layer(x))
            else:
                x = layer(x)
        return x


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size,
        padding,
        bias,
        norm,
        activ=None,
        activ_kwargs={"inplace": True},
    ):
        if activ:
            activ = activ(**activ_kwargs)
        else:
            nn.Identity()
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, bias=bias),
            norm(out_c),
            activ,
        )
        return


class SubMConvNormAct(spconv.SparseSequential):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size=3,
        padding=1,
        bias=False,
        norm=nn.BatchNorm1d,
        activ=nn.ReLU,
    ):
        super().__init__(
            spconv.SubMConv2d(
                in_c,
                out_c,
                kernel_size,
                padding=padding,
                groups=1,
                bias=bias,
                algo=algo,
            ),
            norm(out_c, momentum=0.1),
            activ(inplace=False),
        )


class SubMConvNormReLU(spconv.SparseSequential):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size=3,
        stride=1,
        groups=1,
        activ=nn.BatchNorm1d,
        is_3d=False,
    ):
        padding = (kernel_size - 1) // 2

        if is_3d:
            layer = spconv.SubMConv3d
        else:
            layer = spconv.SubMConv2d
        super(SubMConvNormReLU, self).__init__(
            layer(
                in_c,
                out_c,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
                algo=algo,
            ),
            activ(out_c, momentum=0.1),
            nn.ReLU(inplace=False),
        )


class SparseUpsamplingAdd(spconv.SparseModule):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size,
        indice_key,
        activ=nn.BatchNorm1d,
        bias=False,
        is_3d=False,
    ):
        super().__init__()
        if is_3d:
            layer = spconv.SparseInverseConv3d
        else:
            layer = spconv.SparseInverseConv2d
        self.upsample_layer = spconv.SparseSequential(
            layer(
                in_c, out_c, kernel_size, indice_key=indice_key, bias=bias, algo=algo
            ),
            activ(out_c, momentum=0.1),
        )
        return

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip
