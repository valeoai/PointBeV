from torch import nn

from pointbev.utils.debug import debug_hook


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_forward_hook(debug_hook)
        return

    def forward(self, x, return_all=False):
        raise NotImplementedError()
