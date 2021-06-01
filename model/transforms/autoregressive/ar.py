import torch
from survae.transforms.bijections import Bijection
from survae.utils import sum_except_batch


class AutoregressiveBijection(Bijection):
    """
    Autoregressive bijection.
    Transforms each input variable with an invertible elementwise bijection,
    conditioned on the previous elements.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.

    Args:
        ar_net: nn.Module, an autoregressive network such that `params = ar_net(x)`.
        scheme: An inversion scheme. E.g. RasterScan from utils.
    """
    def __init__(self, ar_net, scheme):
        super(AutoregressiveBijection, self).__init__()
        self.ar_net = ar_net
        self.scheme = scheme
        self.scheme.setup(ar_net=self.ar_net,
                          element_inverse_fn=self._element_inverse)

    def forward(self, x):
        params = self.ar_net(x)
        z, ldj = self._forward(x, params)
        return z, ldj

    def inverse(self, z):
        return self.scheme.inverse(z=z)

    def _num_params(self):
        raise NotImplementedError()

    def _forward(self, x, params):
        raise NotImplementedError()

    def _element_inverse(self, z, element_params):
        raise NotImplementedError()
