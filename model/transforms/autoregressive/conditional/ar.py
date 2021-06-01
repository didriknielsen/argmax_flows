import torch
from survae.transforms.bijections.conditional import ConditionalBijection


class ConditionalAutoregressiveBijection(ConditionalBijection):
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
        super(ConditionalAutoregressiveBijection, self).__init__()
        self.ar_net = ar_net
        self.scheme = scheme
        self.scheme.setup(ar_net=self.ar_net,
                          element_inverse_fn=self._element_inverse)

    def forward(self, x, context):
        params = self.ar_net(x, context=context)
        z, ldj = self._forward(x, params)
        return z, ldj

    def inverse(self, z, context):
        return self.scheme.inverse(z=z, context=context)

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _forward(self, x, params):
        raise NotImplementedError()

    def _element_inverse(self, z, element_params):
        raise NotImplementedError()
