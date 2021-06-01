import torch
from survae.transforms.bijections import Bijection


class Squeeze1d(Bijection):
    """
    A bijection defined for sequential data that trades spatial dimensions for
    channel dimensions, i.e. "squeezes" the inputs along the channel dimensions.
    Introduced in the RealNVP paper [1].
    Args:
        factor: int, the factor to squeeze by (default=2).
        ordered: bool, if True, squeezing happens sequencewise.
                       if False, squeezing happens channelwise.
                       For more details, see example (default=False).
    Source implementation:
        Based on `squeeze_nxn`, `squeeze_2x2`, `squeeze_2x2_ordered`, `unsqueeze_2x2` in:
        https://github.com/laurent-dinh/models/blob/master/real_nvp/real_nvp_utils.py
    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    """

    def __init__(self, factor=2, ordered=False):
        super(Squeeze1d, self).__init__()
        assert isinstance(factor, int)
        assert factor > 1
        self.factor = factor
        self.ordered = ordered

    def _squeeze(self, x):
        assert len(x.shape) == 3, 'Dimension should be 3, but was {}'.format(len(x.shape))
        batch_size, c, l = x.shape
        assert l % self.factor == 0, 'l = {} not multiplicative of {}'.format(l, self.factor)
        t = x.view(batch_size, c, l // self.factor, self.factor)
        if not self.ordered:
            t = t.permute(0, 1, 3, 2).contiguous()
        else:
            t = t.permute(0, 3, 1, 2).contiguous()
        z = t.view(batch_size, c * self.factor, l // self.factor)
        return z

    def _unsqueeze(self, z):
        assert len(z.shape) == 3, 'Dimension should be 3, but was {}'.format(len(z.shape))
        batch_size, c, l = z.shape
        assert c % self.factor == 0, 'c = {} not multiplicative of {}'.format(c, self.factor)
        if not self.ordered:
            t = z.view(batch_size, c // self.factor, self.factor, l)
            t = t.permute(0, 1, 3, 2).contiguous()
        else:
            t = z.view(batch_size, self.factor, c // self.factor, l)
            t = t.permute(0, 2, 3, 1).contiguous()
        x = t.view(batch_size, c // self.factor, l * self.factor)
        return x

    def forward(self, x):
        z = self._squeeze(x)
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        x = self._unsqueeze(z)
        return x
