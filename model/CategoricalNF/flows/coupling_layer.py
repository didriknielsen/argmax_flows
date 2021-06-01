import torch
import torch.nn as nn
import math
import sys
sys.path.append("../../")
from ..flows.flow_layer import FlowLayer
from ..networks.help_layers import run_sequential_with_mask


class CouplingLayer(FlowLayer):

	def __init__(self, c_in, mask, 
					   model_func,
					   block_type=None,
					   c_out=-1,
					   **kwargs):
		super().__init__()
		self.c_in = c_in
		self.c_out = c_out if c_out > 0 else 2 * c_in
		self.register_buffer('mask', mask)
		self.block_type = block_type

		# Scaling factor
		self.scaling_factor = nn.Parameter(torch.zeros(c_in))
		self.nn = model_func(c_out=self.c_out)


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

		nn_out = nn_out.view(nn_out.shape[:-1] + (nn_out.shape[-1]//2, 2))
		s, t = nn_out[...,0], nn_out[...,1]
		
		scaling_fac = self.scaling_factor.exp().view([1, 1, s.size(-1)])
		s = torch.tanh(s / scaling_fac.clamp(min=1.0)) * scaling_fac
		
		s = s * (1 - mask)
		t = t * (1 - mask)

		z, layer_ldj = CouplingLayer.run_with_params(z, s, t, reverse=reverse)
		ldj = ldj + layer_ldj

		return z, ldj # , detail_dict

	def _prepare_mask(self, mask, z):
		# Mask input so that we only use the un-masked regions as input
		mask = self.mask.unsqueeze(dim=0) if len(z.shape) > len(self.mask.shape) else self.mask
		if mask.size(1) < z.size(1) and mask.size(1) > 1:
			mask = mask.repeat(1, int(math.ceil(z.size(1)/mask.size(1))), 1).contiguous()
		if mask.size(1) > z.size(1):
			mask = mask[:,:z.size(1)]
		return mask

	@staticmethod
	def get_coup_params(nn_out, mask, scaling_factor=None):
		nn_out = nn_out.view(nn_out.shape[:-1] + (nn_out.shape[-1]//2, 2))
		s, t = nn_out[...,0], nn_out[...,1]
		if scaling_factor is not None:
			scaling_fac = scaling_factor.exp().view([1, 1, s.size(-1)])
			s = torch.tanh(s / scaling_fac.clamp(min=1.0)) * scaling_fac
		
		s = s * (1 - mask)
		t = t * (1 - mask)
		return s, t

	@staticmethod
	def run_with_params(orig_z, s, t, reverse=False):
		if not reverse:
			scale = torch.exp(s)
			z_out = (orig_z + t) * scale
			ldj = s.sum(dim=[1,2])
		else:
			inv_scale = torch.exp(-1 * s)
			z_out = orig_z * inv_scale - t
			ldj = -s.sum(dim=[1,2])
		return z_out, ldj


	@staticmethod
	def create_channel_mask(c_in, ratio=0.5, mask_floor=True):
		"""
		Ratio: number of channels that are alternated/for which we predict parameters
		"""
		if mask_floor:
			c_masked = int(math.floor(c_in * ratio))
		else:
			c_masked = int(math.ceil(c_in * ratio))
		c_unmasked = c_in - c_masked
		mask = torch.cat([torch.ones(1, c_masked), torch.zeros(1, c_unmasked)], dim=1)
		return mask


	@staticmethod
	def create_chess_mask(seq_len=2):
		assert seq_len > 1
		seq_unmask = int(seq_len // 2)
		seq_mask = seq_len - seq_unmask
		mask = torch.cat([torch.ones(seq_mask, 1), torch.zeros(seq_unmask, 1)], dim=1).view(-1, 1)
		return mask


	def info(self):
		is_channel_mask = (self.mask.size(0) == 1)
		info_str = "Coupling Layer - Input size %i" % (self.c_in)
		if self.block_type is not None:
			info_str += ", block type %s" % (self.block_type)
		info_str += ", mask ratio %.2f, %s mask" % ((1-self.mask).mean().item(), "channel" if is_channel_mask else "chess")
		return info_str