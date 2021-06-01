import torch
import torch.nn as nn
from survae.utils import sum_except_batch
from survae.transforms.bijections.functional import splines
from survae.nn.layers import LambdaLayer
from survae.nn.layers.autoregressive import AutoregressiveShift
from ..transforms.autoregressive.conditional import ConditionalAutoregressiveBijection
from ..transforms.autoregressive.utils import InvertSequentialCL


class ConditionalSplineAutoregressive1d(ConditionalAutoregressiveBijection):

    def __init__(self, c, num_layers, hidden_size, dropout, num_bins, context_size, unconstrained):
        self.unconstrained = unconstrained
        self.num_bins = num_bins
        scheme = InvertSequentialCL(order='cl')
        lstm = ConditionalAutoregressiveLSTM(C=c, P=self._num_params(),
                                             num_layers=num_layers,
                                             hidden_size=hidden_size,
                                             dropout=dropout,
                                             context_size=context_size)
        super(ConditionalSplineAutoregressive1d, self).__init__(ar_net=lstm, scheme=scheme)
        self.register_buffer('constant', torch.log(torch.exp(torch.ones(1)) - 1))
        self.autoregressive_net = self.ar_net # For backwards compatability

    def _num_params(self):
        # return 3 * self.num_bins + 1
        return 3 * self.num_bins - 1

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


class ConditionalAutoregressiveLSTM(nn.Module):

    def __init__(self, C, P, num_layers, hidden_size, dropout, context_size):
        super(ConditionalAutoregressiveLSTM, self).__init__()

        self.l_in = LambdaLayer(lambda x: x.permute(2,0,1)) # (B,C,L) -> (L,B,C)
        self.lstm = ConditionalLayerLSTM(C+context_size, hidden_size, num_layers=num_layers, dropout=dropout) # (L,B,C) -> (L,B,H)
        self.l_out = nn.Sequential(nn.Linear(hidden_size, P*C), # (L,B,H) -> (L,B,P*C)
                                   AutoregressiveShift(P*C),
                                   LambdaLayer(lambda x: x.reshape(*x.shape[0:2], C, P)), # (L,B,P*C) -> (L,B,C,P)
                                   LambdaLayer(lambda x: x.permute(1,2,0,3))) # (L,B,C,P) -> (B,C,L,P)


    def forward(self, x, context):
        x = self.l_in(x)
        context = self.l_in(context)

        x = self.lstm(x, context=context)
        return self.l_out(x)


class ConditionalLayerLSTM(nn.LSTM):

    def forward(self, x, context):
        output, _ = super(ConditionalLayerLSTM, self).forward(torch.cat([x, context], dim=-1)) # output, (c_n, h_n)
        return output
