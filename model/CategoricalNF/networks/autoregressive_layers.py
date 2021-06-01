import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import sys
sys.path.append("../../")

from ..general.mutils import create_T_one_hot
from .help_layers import run_padded_LSTM


class InputDropout(nn.Module):
	"""
	Removes input vectors with a probability of inp_dp_rate
	"""

	def __init__(self, dp_rate=0.0, scale_features=False):
		super().__init__()
		self.dp_rate = dp_rate
		self.scale_features = scale_features

	def forward(self, x):
		if not self.training:
			return x
		else:
			dp_mask = x.new_zeros(x.size(0), x.size(1), 1)
			dp_mask.bernoulli_(p=self.dp_rate)
			x = x * (1 - dp_mask)

			if self.scale_features:
				x = x * 1.0 / (1.0 - self.dp_rate)
			return x


class TimeConcat(nn.Module):

	def __init__(self, time_embed, input_dp_rate=0.0):
		super().__init__()
		self.time_embed_layer = time_embed

		print(f'Using with input dropout: {input_dp_rate}')
		self.input_dropout = InputDropout(input_dp_rate)

	def forward(self, x, time_embed=None, length_one_hot=None, length=None):
		if time_embed is None:
			if length_one_hot is None and length is not None:
				length_one_hot = create_T_one_hot(length, dataset_max_len=int(self.time_embed_layer.weight.data.shape[1]//2))
			time_embed = self.time_embed_layer(length_one_hot)
		x = self.input_dropout(x)
		return torch.cat([x, time_embed], dim=-1)


class LSTMFeatureModel(nn.Module):

	def __init__(self, c_in, c_out, hidden_size, max_seq_len,
					   num_layers=1, dp_rate=0.0, 
					   input_dp_rate=0.0, **kwargs):
		super().__init__()

		time_embed = nn.Linear(2*max_seq_len, int(hidden_size//8))
		time_embed_dim = time_embed.weight.data.shape[0]
		self.time_concat = TimeConcat(time_embed=time_embed, 
									  input_dp_rate=input_dp_rate)
		inp_embed_dim = hidden_size//2 - time_embed_dim
		self.input_embed = nn.Sequential(
				nn.Linear(c_in, hidden_size//2),
				nn.GELU(),
				nn.Linear(hidden_size//2, inp_embed_dim),
				nn.GELU()
			)

		self.lstm_module = nn.LSTM(input_size=inp_embed_dim + time_embed_dim, hidden_size=hidden_size,
									num_layers=num_layers, batch_first=True,
									bidirectional=False, dropout=0.0)
		self.out_layer = AutoregFeedforward(c_in=c_in, c_out_per_in=int(c_out/c_in), 
									   hidden_size=hidden_size//2, c_offset=0)
		self.net = nn.Sequential(
				nn.Dropout(dp_rate),
				nn.Linear(hidden_size, hidden_size//2),
				nn.GELU(),
				nn.Dropout(dp_rate)
			)

	def forward(self, x, length=None, channel_padding_mask=None, length_one_hot=None, **kwargs):
		_inp_embed = self.input_embed(x)
		embed = self.time_concat(x=_inp_embed, length_one_hot=length_one_hot, length=length)
		embed = torch.cat([embed.new_zeros(embed.size(0),1,embed.size(2)), embed[:,:-1]], dim=1)

		lstm_out = run_padded_LSTM(x=embed, lstm_cell=self.lstm_module, length=length)

		out = self.net(lstm_out)
		out = self.out_layer(features=out, _inps=x)
		if channel_padding_mask is not None:
			out = out * channel_padding_mask
		return out


class AutoregressiveLSTMModel(nn.Module):

	def __init__(self, c_in, c_out, hidden_size, max_seq_len, 
					   num_layers=1, dp_rate=0.0, 
					   input_dp_rate=0.0,
					   direction=0, **kwargs):
		super().__init__()
		self.lstm_model = LSTMFeatureModel(c_in, c_out, hidden_size, num_layers=num_layers,
										max_seq_len=max_seq_len,
										dp_rate=dp_rate, input_dp_rate=input_dp_rate)
		self.reverse = (direction == 1)

	def forward(self, x, length=None, channel_padding_mask=None, **kwargs):
		if self.reverse:
			x = self._reverse_input(x, length, channel_padding_mask)
		x = self.lstm_model(x, length=length, channel_padding_mask=channel_padding_mask, shift_by_one=True, **kwargs)
		if self.reverse:
			x = self._reverse_input(x, length, channel_padding_mask)
		return x

	def _reverse_input(self, x, length, channel_padding_mask):
		max_batch_len = x.size(1)
		time_range = torch.arange(max_batch_len, device=length.device).view(1, max_batch_len).expand(length.size(0),-1).long()
		indices = ((length.unsqueeze(dim=1)-1) - time_range).clamp(min=0)
		indices = indices.unsqueeze(dim=-1).expand(-1, -1, x.size(2))
		x_inv = x.gather(index=indices, dim=1)
		x_inv = x_inv * channel_padding_mask
		return x_inv


class AutoregFeedforward(nn.Module):

	def __init__ (self, c_in, c_out_per_in, hidden_size, c_offset=0):
		super().__init__()
		self.c_in = c_in
		self.c_autoreg = c_in - 1 - c_offset
		self.c_out_per_in = c_out_per_in
		self.c_offset = c_offset
		self.hidden_size = hidden_size
		self.embed_size = min(max(1, int(hidden_size*9.0/16.0/(self.c_in-1))), 96)
		self.hidden_dim_2 = int(hidden_size//2)
		self.act_fn_1 = nn.GELU()
		self.act_fn_2 = nn.GELU()
		self.in_to_features = nn.Linear((self.c_in-1)*3, self.embed_size*(self.c_in-1))
		self.features_to_hidden = nn.Linear(hidden_size + self.embed_size*(self.c_in-1), self.hidden_dim_2*self.c_in)
		self.hidden_to_out = nn.Linear(self.hidden_dim_2*self.c_in, c_out_per_in*self.c_in)
		mask_in_to_features, mask_features_to_hidden, mask_hidden_to_out = self._create_masks()
		self.register_buffer("mask_in_to_features", mask_in_to_features)
		self.register_buffer("mask_features_to_hidden", mask_features_to_hidden)
		self.register_buffer("mask_hidden_to_out", mask_hidden_to_out)

	def forward(self, features, _inps):
		self._mask_layers()
		if _inps.size(-1) == self.c_in:
			_inps = _inps[...,:-1] # The last channel is not used as input for any transformation
		_inps = torch.stack([_inps, F.elu(_inps), F.elu(-_inps)], dim=-1).view(_inps.shape[:-1]+(3*_inps.shape[-1],))
		in_features = self.in_to_features(_inps)
		in_features = self.act_fn_1(in_features)
		features = torch.cat([features, in_features], dim=-1)
		hidden = self.features_to_hidden(features)
		hidden = self.act_fn_2(hidden)
		out = self.hidden_to_out(hidden)
		return out

	def _create_masks(self):
		mask_in_to_features = torch.ones_like(self.in_to_features.weight.data) # [self.embed_size*(c_in-1), c_in-1]
		for c_in in range(self.c_offset, self.c_in-1):
			mask_in_to_features[:self.embed_size*c_in, c_in*3:(c_in+1)*3] = 0
			mask_in_to_features[self.embed_size*(c_in+1):, c_in*3:(c_in+1)*3] = 0
			mask_in_to_features[self.embed_size*c_in:self.embed_size*(c_in+1), :c_in*3] = 0
			mask_in_to_features[self.embed_size*c_in:self.embed_size*(c_in+1), (c_in+1)*3:] = 0

		mask_features_to_hidden = torch.ones_like(self.features_to_hidden.weight.data) # [self.hidden_dim_2*c_in, hidden_size + self.embed_size*(c_in-1)]
		for c_in in range(self.c_in):
			mask_features_to_hidden[self.hidden_dim_2*c_in:self.hidden_dim_2*(c_in+1), self.hidden_size+self.embed_size*(self.c_offset + max(0,c_in-self.c_offset)):] = 0
		
		mask_hidden_to_out = torch.ones_like(self.hidden_to_out.weight.data) # [c_out_per_in*c_in, self.hidden_dim_2*c_in]
		for c_in in range(self.c_in):
			mask_hidden_to_out[:self.c_out_per_in*c_in, self.hidden_dim_2*c_in:self.hidden_dim_2*(c_in+1)] = 0
			mask_hidden_to_out[self.c_out_per_in*(c_in+1):, self.hidden_dim_2*c_in:self.hidden_dim_2*(c_in+1)] = 0

		return mask_in_to_features, mask_features_to_hidden, mask_hidden_to_out

	def _mask_layers(self):
		self.in_to_features.weight.data *= self.mask_in_to_features
		self.features_to_hidden.weight.data *= self.mask_features_to_hidden
		self.hidden_to_out.weight.data *= self.mask_hidden_to_out