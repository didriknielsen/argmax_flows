
import math
import torch
import torch.nn as nn


class MaskedLinear(nn.Linear):

    def __init__(self, in_dim, out_dim, data_dim, causal, bias=True):
        super(MaskedLinear, self).__init__(in_features=in_dim, out_features=out_dim, bias=bias)
        self.register_buffer('mask', self.create_mask(in_dim, out_dim, data_dim, causal))
        self.data_dim = data_dim
        self.causal = causal

    @staticmethod
    def create_mask(in_dim, out_dim, data_dim, causal):
        base_mask = torch.ones([data_dim,data_dim])
        if causal: base_mask = base_mask.tril(-1)
        else:      base_mask = base_mask.tril(0)
        rep_out, rep_in = math.ceil(out_dim / data_dim), math.ceil(in_dim / data_dim)
        return base_mask.repeat(rep_out, rep_in)[0:out_dim, 0:in_dim]

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedLinear, self).forward(x)
