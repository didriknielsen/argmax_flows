import torch
import numpy as np
import torch.nn.functional as F
from survae.distributions import ConditionalDistribution
from survae.transforms.surjections import Surjection
from .utils import integer_to_base, base_to_integer


class BinaryProductArgmaxSurjection(Surjection):
    '''
    A generative argmax surjection using a Cartesian product of binary spaces. Argmax is performed over the final dimension.
    Args:
        encoder: ConditionalDistribution, a distribution q(z|x) with support over z s.t. x=argmax z.
    Example:
        Input tensor x of shape (B, D, L) with discrete values {0,1,...,C-1}:
        encoder should be a distribution of shape (B, D, L, D), where D=ceil(log2(C)).
        When e.g. C=27, we have D=5, such that 2**5=32 classes are represented.
    '''
    stochastic_forward = True

    def __init__(self, encoder, num_classes):
        super(BinaryProductArgmaxSurjection, self).__init__()
        assert isinstance(encoder, ConditionalDistribution)
        self.encoder = encoder
        self.num_classes = num_classes
        self.dims = self.classes2dims(num_classes)

    @staticmethod
    def classes2dims(num_classes):
        return int(np.ceil(np.log2(num_classes)))

    def idx2base(self, idx_tensor):
        return integer_to_base(idx_tensor, base=2, dims=self.dims)

    def base2idx(self, base_tensor):
        return base_to_integer(base_tensor, base=2)

    def forward(self, x):
        z, log_qz = self.encoder.sample_with_log_prob(context=x)
        ldj = -log_qz
        return z, ldj

    def inverse(self, z):
        binary = torch.gt(z, 0.0).long()
        idx = self.base2idx(binary)
        return idx
