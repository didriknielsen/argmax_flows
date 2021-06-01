import torch
import torch.nn as nn
from survae.utils import sum_except_batch
from survae.transforms.bijections.functional import splines
from ..transforms.autoregressive.conditional import ConditionalAutoregressiveBijection
from ..transforms.autoregressive.utils import InvertSequentialCL


class CondSplineAR(ConditionalAutoregressiveBijection):

    def __init__(self, ar_net, num_bins, unconstrained):
        self.unconstrained = unconstrained
        self.num_bins = num_bins
        scheme = InvertSequentialCL(order='cl')
        super(CondSplineAR, self).__init__(ar_net=ar_net, scheme=scheme)
        self.register_buffer('constant', torch.log(torch.exp(torch.ones(1)) - 1))

    def _num_params(self):
        return 3 * self.num_bins + 1

    def _forward(self, x, params):
        unnormalized_widths = params[..., :self.num_bins]
        unnormalized_heights = params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = params[..., 2*self.num_bins:] + self.constant
        if self.unconstrained:
            z, ldj_elementwise = splines.unconstrained_rational_quadratic_spline(
                x,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=False)
        else:
            z, ldj_elementwise = splines.rational_quadratic_spline(
                x,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=False)

        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _element_inverse(self, z, element_params):
        unnormalized_widths = element_params[..., :self.num_bins]
        unnormalized_heights = element_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = element_params[..., 2*self.num_bins:] + self.constant
        if self.unconstrained:
            x, _ = splines.unconstrained_rational_quadratic_spline(
                z,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=True)
        else:
            x, _ = splines.rational_quadratic_spline(
                z,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=True)
        return x
