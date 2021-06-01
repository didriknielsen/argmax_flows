import torch
import torch.nn as nn
from survae.utils import sum_except_batch
from ..transforms.autoregressive.conditional import ConditionalAffineAutoregressiveBijection
from ..transforms.autoregressive.utils import InvertSequentialCL


class CondAffineAR(ConditionalAffineAutoregressiveBijection):

    def __init__(self, ar_net):
        scheme = InvertSequentialCL(order='cl')
        super(CondAffineAR, self).__init__(ar_net, scheme=scheme)

    def _forward(self, x, params):
        unconstrained_scale, shift = self._split_params(params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        z = shift + torch.exp(log_scale) * x
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _element_inverse(self, z, element_params):
        unconstrained_scale, shift = self._split_params(params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        x = (z - shift) * torch.exp(-log_scale)
        return x
