import torch
import torch.nn as nn
import torch.nn.functional as F 
import sys
sys.path.append("../../")

from general.mutils import one_hot
from layers.flows.coupling_layer import CouplingLayer


class DiscreteCouplingLayer(CouplingLayer):

	def __init__(self, c_in, mask, 
					   model_func,
					   block_type=None,
					   temp=0.1,
					   **kwargs):
		super().__init__(c_in=c_in,
						 mask=mask,
						 model_func=model_func,
						 block_type=block_type,
						 c_out=c_in)
		self.temp = temp
		del self.scaling_factor


	def run_network(self, x, length=None, **kwargs):
		if isinstance(self.nn, nn.Sequential):
			nn_out = run_sequential_with_mask(self.nn, x, 
											  length=length, 
											  **kwargs)
		else:
			nn_out = self.nn(x, length=length,
							 **kwargs)

		if "channel_padding_mask" in kwargs and kwargs["channel_padding_mask"] is not None:
			nn_out = nn_out * kwargs["channel_padding_mask"]
		return nn_out


	def forward(self, z, ldj=None, reverse=False, channel_padding_mask=None, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0),)
		if channel_padding_mask is None:
			channel_padding_mask = torch.ones_like(z)

		mask = self._prepare_mask(self.mask, z)
		z_in = z * mask
		nn_out = self.run_network(x=z_in, **kwargs)
		shift_one_hot = one_hot_argmax(logits=nn_out, temperature=self.temp)
		default_shift = torch.zeros_like(shift_one_hot)
		default_shift[...,0] = 1 # Does not shift output
		shift_one_hot = shift_one_hot * (1 - mask) + default_shift * mask

		z = one_hot_add(z, shift_one_hot, reverse=reverse)
		return z, ldj


	def info(self):
		info_str = "Discrete Coupling Layer - Input size %i" % (self.c_in)
		if self.block_type is not None:
			info_str += ", block type %s" % (self.block_type)
		info_str += ", temperature %.1f" % (self.temp)
		return info_str


def one_hot_argmax(logits, temperature=1.0):
	probs = F.softmax(logits, dim=-1)
	one_hot_argmax = one_hot(probs.argmax(dim=-1), num_classes=probs.size(-1))
	one_hot_approx = (one_hot_argmax - probs).detach() + probs
	return one_hot_approx


def one_hot_add(inp_one_hot, shift_one_hot, reverse=False):
	num_categ = inp_one_hot.size(-1)
	roll_matrix = torch.stack([torch.roll(input=shift_one_hot, shifts=i, dims=-1) for i in range(num_categ)], dim=-2)
	if reverse:
		roll_matrix = torch.transpose(roll_matrix, dim0=-2, dim1=-1)
	inp_one_hot = inp_one_hot.unsqueeze(dim=-2)
	out_one_hot = torch.matmul(inp_one_hot, roll_matrix)
	out_one_hot = out_one_hot.squeeze(dim=-2)
	return out_one_hot