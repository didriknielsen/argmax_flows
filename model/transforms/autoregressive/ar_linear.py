import torch
from survae.utils import sum_except_batch
from .ar import AutoregressiveBijection


class AdditiveAutoregressiveBijection(AutoregressiveBijection):
    '''Additive autoregressive bijection.'''

    def _num_params(self):
        return 1

    def _forward(self, x, params):
        return x + params, x.new_zeros(x.shape[0])

    def _element_inverse(self, z, element_params):
        return z - element_params


class AffineAutoregressiveBijection(AutoregressiveBijection):
    '''Affine autoregressive bijection.'''

    def _num_params(self):
        return 2

    def _forward(self, x, params):
        assert params.shape[-1] == self._num_params()
        log_scale, shift = self._split_params(params)
        scale = torch.exp(log_scale)
        z = scale * x + shift
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _element_inverse(self, z, element_params):
        assert element_params.shape[-1] == self._num_params()
        log_scale, shift = self._split_params(element_params)
        scale = torch.exp(log_scale)
        x = (z - shift) / scale
        return x

    def _split_params(self, params):
        unconstrained_scale = params[..., 0]
        shift = params[..., 1]
        return unconstrained_scale, shift
