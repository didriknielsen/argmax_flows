import torch
from survae.distributions import DiagonalNormal


class ConvNormal1d(DiagonalNormal):
    def __init__(self, shape):
        super(DiagonalNormal, self).__init__()
        assert len(shape) == 2
        self.shape = torch.Size(shape)
        self.loc = torch.nn.Parameter(torch.zeros(1, shape[0], 1))
        self.log_scale = torch.nn.Parameter(torch.zeros(1, shape[0], 1))
