import torch
from survae.distributions import Distribution
from survae.utils import sum_except_batch


class StandardGumbel(Distribution):
    """A standard Gumbel distribution."""

    def __init__(self, shape):
        super(StandardGumbel, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))

    def log_prob(self, x):
        return sum_except_batch(- x - (-x).exp())

    def sample(self, num_samples):
        u = torch.rand(num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype)
        eps = torch.finfo(u.dtype).tiny # 1.18e-38 for float32
        return -torch.log(-torch.log(u + eps) + eps)
