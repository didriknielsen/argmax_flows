import torch
import torch.nn as nn

from survae.flows import Flow, ConditionalInverseFlow
from survae.distributions import StandardUniform, StandardNormal, ConditionalNormal
from survae.transforms import ActNormBijection1d, PermuteAxes, Reshape, Shuffle, Conv1x1, Reverse, Softplus
from ..transforms import BinaryProductArgmaxSurjection, Squeeze1d
from ..distributions import StandardGumbel, ConvNormal1d, BinaryEncoder
from .encoder_context import IdxContextNet
from .encoder_transforms import ConditionalSplineAutoregressive1d

from ..CategoricalNF.flows.autoregressive_coupling import \
    AutoregressiveMixtureCDFCoupling
from ..CategoricalNF.networks.autoregressive_layers import \
    AutoregressiveLSTMModel

def ar_func(c_in, c_out, hidden, num_layers, max_seq_len, input_dp_rate):
    return AutoregressiveLSTMModel(
        c_in=c_in,
        c_out=c_out,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        hidden_size=hidden,
        dp_rate=0,
        input_dp_rate=input_dp_rate)


class ArgmaxARFlow(Flow):

    def __init__(self, data_shape, num_classes,
                 num_steps, actnorm, perm_channel, perm_length, base_dist,
                 encoder_steps, encoder_bins, context_size,
                 lstm_layers, lstm_size, lstm_dropout,
                 context_lstm_layers, context_lstm_size, input_dp_rate):

        transforms = []
        C, L = data_shape
        K = BinaryProductArgmaxSurjection.classes2dims(num_classes)

        # Encoder context
        context_net = IdxContextNet(num_classes=num_classes,
                                    context_size=context_size,
                                    num_layers=context_lstm_layers,
                                    hidden_size=context_lstm_size,
                                    dropout=lstm_dropout)

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

            encoder_transforms.append(ConditionalSplineAutoregressive1d(c=encoder_shape[0],
                                                                        num_layers=lstm_layers,
                                                                        hidden_size=lstm_size,
                                                                        dropout=lstm_dropout,
                                                                        num_bins=encoder_bins,
                                                                        context_size=context_size,
                                                                        unconstrained=True))
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

            def model_func(c_out):
                return ar_func(
                    c_in=current_shape[0],
                    c_out=c_out,
                    hidden=lstm_size,
                    num_layers=lstm_layers,
                    max_seq_len=L,
                    input_dp_rate=input_dp_rate)

            transforms.append(
                AutoregressiveMixtureCDFCoupling(
                    c_in=current_shape[0],
                    model_func=model_func,
                    block_type="LSTM model",
                    num_mixtures=27)
            )

        if base_dist == 'conv_gauss': base_dist = ConvNormal1d(current_shape)
        elif base_dist == 'gauss':    base_dist = StandardNormal(current_shape)
        elif base_dist == 'gumbel':   base_dist = StandardGumbel(current_shape)
        super(ArgmaxARFlow, self).__init__(base_dist=base_dist,
                                           transforms=transforms)
