import torch
from collections.abc import Iterable
from survae.utils import sum_except_batch
from survae.transforms.bijections.functional import splines
from .ar import AutoregressiveBijection


class LinearSplineAutoregressiveBijection(AutoregressiveBijection):

    def __init__(self, ar_net, scheme, num_bins):
        super(LinearSplineAutoregressiveBijection, self).__init__(ar_net=ar_net, scheme=scheme)
        self.num_bins = num_bins

    def _num_params(self):
        return self.num_bins

    def _forward(self, x, params):
        assert params.shape[-1] == self._num_params()
        z, ldj_elementwise = splines.linear_spline(x, params, inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _element_inverse(self, z, element_params):
        assert element_params.shape[-1] == self._num_params()
        x, _ = splines.linear_spline(z, element_params, inverse=True)
        return x


class QuadraticSplineAutoregressiveBijection(AutoregressiveBijection):

    def __init__(self, ar_net, scheme, num_bins):
        super(QuadraticSplineAutoregressiveBijection, self).__init__(ar_net=ar_net, scheme=scheme)
        self.num_bins = num_bins

    def _num_params(self):
        return 2 * self.num_bins + 1

    def _forward(self, x, params):
        assert params.shape[-1] == self._num_params()
        unnormalized_widths, unnormalized_heights = params[..., :self.num_bins], params[..., self.num_bins:]
        z, ldj_elementwise = splines.quadratic_spline(x, unnormalized_widths=unnormalized_widths, unnormalized_heights=unnormalized_heights, inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _element_inverse(self, z, element_params):
        assert element_params.shape[-1] == self._num_params()
        unnormalized_widths, unnormalized_heights = element_params[..., :self.num_bins], element_params[..., self.num_bins:]
        x, _ = splines.quadratic_spline(z, unnormalized_widths=unnormalized_widths, unnormalized_heights=unnormalized_heights, inverse=True)
        return x


class CubicSplineAutoregressiveBijection(AutoregressiveBijection):

    def __init__(self, ar_net, scheme, num_bins):
        super(CubicSplineAutoregressiveBijection, self).__init__(ar_net=ar_net, scheme=scheme)
        self.num_bins = num_bins

    def _num_params(self):
        return 2 * self.num_bins + 2

    def _forward(self, x, params):
        assert params.shape[-1] == self._num_params()
        unnormalized_widths = params[..., :self.num_bins]
        unnormalized_heights = params[..., self.num_bins:2*self.num_bins]
        unnorm_derivatives_left = params[..., 2*self.num_bins:2*self.num_bins+1]
        unnorm_derivatives_right = params[..., 2*self.num_bins+1:]
        z, ldj_elementwise = splines.cubic_spline(x,
                                                  unnormalized_widths=unnormalized_widths,
                                                  unnormalized_heights=unnormalized_heights,
                                                  unnorm_derivatives_left=unnorm_derivatives_left,
                                                  unnorm_derivatives_right=unnorm_derivatives_right,
                                                  inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _element_inverse(self, z, element_params):
        assert element_params.shape[-1] == self._num_params()
        unnormalized_widths = element_params[..., :self.num_bins]
        unnormalized_heights = element_params[..., self.num_bins:2*self.num_bins]
        unnorm_derivatives_left = element_params[..., 2*self.num_bins:2*self.num_bins+1]
        unnorm_derivatives_right = element_params[..., 2*self.num_bins+1:]
        x, _ = splines.cubic_spline(z,
                                    unnormalized_widths=unnormalized_widths,
                                    unnormalized_heights=unnormalized_heights,
                                    unnorm_derivatives_left=unnorm_derivatives_left,
                                    unnorm_derivatives_right=unnorm_derivatives_right,
                                    inverse=True)
        return x


class RationalQuadraticSplineAutoregressiveBijection(AutoregressiveBijection):

    def __init__(self, ar_net, scheme, num_bins):
        super(RationalQuadraticSplineAutoregressiveBijection, self).__init__(ar_net=ar_net, scheme=scheme)
        self.num_bins = num_bins

    def _num_params(self):
        return 3 * self.num_bins + 1

    def _forward(self, x, params):
        assert params.shape[-1] == self._num_params()
        unnormalized_widths = params[..., :self.num_bins]
        unnormalized_heights = params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = params[..., 2*self.num_bins:]
        z, ldj_elementwise = splines.rational_quadratic_spline(x,
                                                               unnormalized_widths=unnormalized_widths,
                                                               unnormalized_heights=unnormalized_heights,
                                                               unnormalized_derivatives=unnormalized_derivatives,
                                                               inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _element_inverse(self, z, element_params):
        assert element_params.shape[-1] == self._num_params()
        unnormalized_widths = element_params[..., :self.num_bins]
        unnormalized_heights = element_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = element_params[..., 2*self.num_bins:]
        x, _ = splines.rational_quadratic_spline(z,
                                                 unnormalized_widths=unnormalized_widths,
                                                 unnormalized_heights=unnormalized_heights,
                                                 unnormalized_derivatives=unnormalized_derivatives,
                                                 inverse=True)
        return x
