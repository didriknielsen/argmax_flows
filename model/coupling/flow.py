import torch
import torch.nn as nn
import torch.nn.functional as F

from survae.flows import Flow, ConditionalInverseFlow
from survae.distributions import StandardUniform, StandardNormal, ConditionalNormal
from survae.transforms import ActNormBijection1d, PermuteAxes, Reshape, Shuffle, Conv1x1, Reverse, Softplus
from survae.nn.layers import LambdaLayer
from ..transforms import BinaryProductArgmaxSurjection, Squeeze1d
from ..distributions import StandardGumbel, ConvNormal1d, BinaryEncoder
from .cond_ar_affine import CondAffineAR
from .cond_ar_spline import CondSplineAR
from .masked_linear import MaskedLinear

from ..CategoricalNF.flows.autoregressive_coupling2 import \
    CouplingMixtureCDFCoupling
from ..CategoricalNF.networks.autoregressive_layers2 import \
    CouplingLSTMModel


class ArgmaxCouplingFlow(Flow):

    def __init__(self, data_shape, num_classes,
                 num_steps, actnorm, num_mixtures,
                 perm_channel, perm_length, base_dist,
                 encoder_steps, encoder_bins, encoder_ff_size,
                 context_size, context_ff_layers, context_ff_size, context_dropout,
                 lstm_layers, lstm_size, lstm_dropout, input_dp_rate):

        transforms = []
        C, L = data_shape
        K = BinaryProductArgmaxSurjection.classes2dims(num_classes)
        E = 2 if encoder_bins is None else 3 * encoder_bins + 1

        # Encoder context
        context_net = ContextFF(num_classes=num_classes,
                                context_size=context_size,
                                num_layers=context_ff_layers,
                                hidden_size=context_ff_size,
                                dropout=context_dropout)

        # Encoder base
        encoder_shape = (C*K, L)
        encoder_base = ConditionalNormal(nn.Conv1d(context_size, 2*C*K, kernel_size=1, padding=0), split_dim=1)

        # Encoder transforms
        encoder_transforms = []
        for step in range(encoder_steps):
            if step > 0:
                if actnorm: encoder_transforms.append(ActNormBijection1d(encoder_shape[0]))
                if perm_length == 'reverse':    encoder_transforms.append(Reverse(encoder_shape[1], dim=2))
                if perm_channel == 'conv':      encoder_transforms.append(Conv1x1(encoder_shape[0], slogdet_cpu=False))
                elif perm_channel == 'shuffle': encoder_transforms.append(Shuffle(encoder_shape[0]))

            encoder_net = ARFF(data_dim=K,
                               context_dim=context_size,
                               num_params=E,
                               hidden_size=encoder_ff_size)

            if encoder_bins is None:
                encoder_transforms.append(CondAffineAR(ar_net=encoder_net))
            else:
                encoder_transforms.append(CondSplineAR(ar_net=encoder_net, num_bins=encoder_bins, unconstrained=True))

        encoder_transforms.append(Reshape((C*K,L), (C,K,L))) # (B,C*K,L) -> (B,C,K,L)
        encoder_transforms.append(PermuteAxes([0,1,3,2])) # (B,C,K,L) -> (B,C,L,K)

        # Encoder
        encoder = BinaryEncoder(ConditionalInverseFlow(base_dist=encoder_base,
                                                       transforms=encoder_transforms,
                                                       context_init=context_net), dims=K)
        transforms.append(BinaryProductArgmaxSurjection(encoder, num_classes))

        # Reshape
        transforms.append(PermuteAxes([0,1,3,2])) # (B,C,L,K) -> (B,C,K,L)
        transforms.append(Reshape((C,K,L), (C*K,L))) # (B,C,K,L) -> (B,C*K,L)
        current_shape = (C*K,L)

        # Coupling blocks
        for step in range(num_steps):
            if step > 0:
                if actnorm: transforms.append(ActNormBijection1d(current_shape[0]))
                if perm_length == 'reverse':    transforms.append(Reverse(current_shape[1], dim=2))
                if perm_channel == 'conv':      transforms.append(Conv1x1(current_shape[0], slogdet_cpu=False))
                elif perm_channel == 'shuffle': transforms.append(Shuffle(current_shape[0]))

            def model_func(c_in, c_out):
                return CouplingLSTMModel(
                    c_in=c_in,
                    c_out=c_out,
                    max_seq_len=L,
                    num_layers=lstm_layers,
                    hidden_size=lstm_size,
                    dp_rate=0,
                    input_dp_rate=input_dp_rate)

            transforms.append(
                CouplingMixtureCDFCoupling(
                    c_in=K//2,
                    c_out=K//2+K%2,
                    model_func=model_func,
                    block_type="LSTM model",
                    num_mixtures=num_mixtures)
            )

        if base_dist == 'conv_gauss': base_dist = ConvNormal1d(current_shape)
        elif base_dist == 'gauss':    base_dist = StandardNormal(current_shape)
        elif base_dist == 'gumbel':   base_dist = StandardGumbel(current_shape)
        super(ArgmaxCouplingFlow, self).__init__(base_dist=base_dist,
                                                 transforms=transforms)


class ARFF(nn.Module):

    def __init__ (self, data_dim, context_dim, num_params, hidden_size):
        super(ARFF, self).__init__()
        self.num_params = num_params
        self.data_dim = data_dim
        self.emb = MaskedLinear(3*data_dim, hidden_size, data_dim=data_dim, causal=True)
        self.cat = MaskedLinear(hidden_size+context_dim, hidden_size, data_dim=data_dim, causal=False)
        self.out = MaskedLinear(hidden_size, data_dim * num_params, data_dim=data_dim, causal=False)
        self.cat.mask[:,hidden_size:] = 1.0

    def forward(self, x, context):
        x = x.permute(0,2,1)
        context = context.permute(0,2,1)
        emb = torch.cat([x, F.elu(x), F.elu(-x)], dim=-1)
        emb = F.gelu(self.emb(emb))
        out = torch.cat([emb, context], dim=-1)
        out = F.gelu(self.cat(out))
        out = self.out(out)
        out = out.reshape(*out.shape[:-1], self.num_params, self.data_dim)
        return out.permute(0,3,1,2)


class ContextFF(nn.Sequential):

    def __init__(self, num_classes, context_size, num_layers, hidden_size, dropout):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())
        super(ContextFF, self).__init__(
            LambdaLayer(lambda x: x.squeeze(1)), # (B,1,L) -> (B,L)
            nn.Embedding(num_classes, hidden_size), # (B,L,H)
            *layers,
            nn.Linear(hidden_size, context_size), # (B,L,H) -> (B,L,P)
            LambdaLayer(lambda x: x.permute(0,2,1))) # (B,L,P) -> (B,P,L)
