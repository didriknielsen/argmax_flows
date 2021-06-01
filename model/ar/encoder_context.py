import torch
import torch.nn as nn
from survae.nn.layers import LambdaLayer
from survae.nn.layers.autoregressive import AutoregressiveShift


class IdxContextNet(nn.Sequential):

    def __init__(self, num_classes, context_size, num_layers, hidden_size, dropout):
        super(IdxContextNet, self).__init__(
            LambdaLayer(lambda x: x.squeeze(1)), # (B,1,L) -> (B,L)
            nn.Embedding(num_classes, hidden_size), # (B,L,H)
            LambdaLayer(lambda x: x.permute(1,0,2)), # (B,L,H) -> (L,B,H)
            LayerLSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True), # (L,B,H) -> (L,B,2*H)
            nn.Linear(2*hidden_size, context_size), # (L,B,2*H) -> (L,B,P)
            # AutoregressiveShift(context_size),
            LambdaLayer(lambda x: x.permute(1,2,0))) # (L,B,P) -> (B,P,L)


class RealContextNet(nn.Sequential):

    def __init__(self, input_channels, context_size, num_layers, hidden_size, dropout):
        super(RealContextNet, self).__init__(
            LambdaLayer(lambda x: x.squeeze()),  # (B,1,L) -> (B,L)
            LambdaLayer(lambda x: x.permute(1, 0, 2)),  # (B,L,H) -> (L,B,H)
            LayerLSTM(input_channels, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True), # (L,B,H) -> (L,B,2*H)
            nn.Linear(2*hidden_size, context_size),  # (L,B,2*H) -> (L,B,P)
            # AutoregressiveShift(context_size),
            LambdaLayer(lambda x: x.permute(1, 2, 0)))  # (L,B,P) -> (B,P,L)


class LayerLSTM(nn.LSTM):
    def forward(self, x):
        output, _ = super(LayerLSTM, self).forward(x) # output, (c_n, h_n)
        return output
