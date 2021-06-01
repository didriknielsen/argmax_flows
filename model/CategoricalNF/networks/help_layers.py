import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import sys
sys.path.append("../../")



class PositionalEmbedding(nn.Module):

	def __init__(self, d_model, max_seq_len):
		super(PositionalEmbedding, self).__init__()

		pos_embed_dim = int(d_model//2)
		self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, pos_embed_dim), requires_grad=True)
		self.comb_layer = nn.Sequential(
				nn.Linear(d_model + pos_embed_dim, d_model),
				nn.ELU(),
				nn.LayerNorm(d_model)
			)

	def forward(self, x):
		x = torch.cat([x, self.pos_emb.expand(x.size(0),-1,-1)], dim=-1)
		out = self.comb_layer(x)
		return out

class IndependentLinear(nn.Module):

	def __init__(self, D, hidden_dim, c_out):
		super().__init__()
		self.layer1 = nn.Linear(D*hidden_dim, D*hidden_dim)
		self.act_fn = nn.GELU()
		self.layer2 = nn.Linear(D*hidden_dim, D*c_out)

		mask_layer1 = torch.zeros_like(self.layer1.weight.data)
		for d in range(D):
			mask_layer1[d*hidden_dim:(d+1)*hidden_dim, d*hidden_dim:(d+1)*hidden_dim] = 1.0
		self.register_buffer("mask_layer1", mask_layer1)

		mask_layer2 = torch.zeros_like(self.layer2.weight.data)
		for d in range(D):
			mask_layer2[d,d*hidden_dim:(d+1)*hidden_dim] = 1.0
		self.register_buffer("mask_layer2", mask_layer2)

	def forward(self, x):
		self.layer1.weight.data.mul_(self.mask_layer1)
		self.layer2.weight.data.mul_(self.mask_layer2)

		x = self.layer1(x)
		x = self.act_fn(x)
		x = self.layer2(x)

		return x


class SimpleLinearLayer(nn.Module):

	def __init__(self, c_in, c_out, data_init=False):
		super().__init__()
		self.layer = nn.Linear(c_in, c_out)
		if data_init:
			scale_dims = int(c_out//2)
			self.layer.weight.data[scale_dims:,:] = 0
			self.layer.weight.data = self.layer.weight.data * 4 / np.sqrt(c_out/2)
			self.layer.bias.data.zero_()

	def forward(self, x, **kwargs):
		return self.layer(x)

	def initialize_zeros(self):
		self.layer.weight.data.zero_()
		self.layer.bias.data.zero_()


class LinearNet(nn.Module):

	def __init__(self, c_in, c_out, num_layers, hidden_size, ext_input_dims=0, zero_init=False):
		super().__init__()
		self.inp_layer = nn.Sequential(
				nn.Linear(c_in, hidden_size),
				nn.GELU()
			)
		self.main_net = []
		for i in range(num_layers):
			self.main_net += [
				nn.Linear(hidden_size if i>0 else hidden_size + ext_input_dims, 
						  hidden_size),
				nn.GELU()
			]
		self.main_net += [
			nn.Linear(hidden_size, c_out)
		]
		self.main_net = nn.Sequential(*self.main_net)
		if zero_init:
			self.main_net[-1].weight.data.zero_()
			self.main_net[-1].bias.data.zero_()

	def forward(self, x, ext_input=None, **kwargs):
		x_feat = self.inp_layer(x)
		if ext_input is not None:
			x_feat = torch.cat([x_feat, ext_input], dim=-1)
		out = self.main_net(x_feat)
		return out

	def set_bias(self, bias):
		self.main_net[-1].bias.data = bias



def run_sequential_with_mask(net, x, length=None, channel_padding_mask=None, src_key_padding_mask=None, length_one_hot=None, time_embed=None, gt=None, importance_weight=1, detail_out=False, **kwargs):
	dict_detail_out = dict()
	if channel_padding_mask is None:
		nn_out = net(x)
	else:
		x = x * channel_padding_mask
		for l in net:
			x = l(x)
		nn_out = x * channel_padding_mask # Making sure to zero out the outputs for all padding symbols

	if not detail_out:
		return nn_out
	else:
		return nn_out, dict_detail_out


def run_padded_LSTM(x, lstm_cell, length, input_memory=None, return_final_states=False):
	if length is not None and (length != x.size(1)).sum() > 0:
		# Sort input elements for efficient LSTM application
		sorted_lengths, perm_index = length.sort(0, descending=True)
		x = x[perm_index]

		packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, sorted_lengths, batch_first=True)
		packed_outputs, _ = lstm_cell(packed_input, input_memory)
		outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

		# Redo sort
		_, unsort_indices = perm_index.sort(0, descending=False)
		outputs = outputs[unsort_indices]
	else:
		outputs, _ = lstm_cell(x, input_memory)
	return outputs


if __name__ == "__main__":
	pass