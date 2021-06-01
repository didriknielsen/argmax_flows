import torch
import torch.nn as nn
import sys
sys.path.append("../../")

from ..flows.flow_layer import FlowLayer


class ActNormFlow(FlowLayer):
	"""
	Normalizes the activations over channels
	"""


	def __init__(self, c_in, data_init=True):
		super().__init__()
		self.c_in = c_in 
		self.data_init = data_init

		self.bias = nn.Parameter(torch.zeros(1, 1, self.c_in))
		self.scales = nn.Parameter(torch.zeros(1, 1, self.c_in))


	def forward(self, z, ldj=None, reverse=False, length=None, channel_padding_mask=None, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0),)
		if length is None:
			if channel_padding_mask is None:
				length = z.size(1)
			else:
				length = channel_padding_mask.squeeze(dim=2).sum(dim=1)
		else:
			length = length.float()
		
		if not reverse:
			z = (z + self.bias) * torch.exp(self.scales)
			ldj += self.scales.sum(dim=[1,2]) * length
		else:
			z = z * torch.exp(-self.scales) - self.bias
			ldj += (-self.scales.sum(dim=[1,2])) * length

		if channel_padding_mask is not None:
			z = z * channel_padding_mask

		assert torch.isnan(z).sum() == 0, "[!] ERROR: z contains NaN values."
		assert torch.isnan(ldj).sum() == 0, "[!] ERROR: ldj contains NaN values."

		return z, ldj


	def need_data_init(self):
		return self.data_init


	def data_init_forward(self, input_data, channel_padding_mask=None, **kwargs):
		if channel_padding_mask is None:
			channel_padding_mask = input_data.new_ones(input_data.shape)
		mask = channel_padding_mask
		num_exp = mask.sum(dim=[0,1], keepdims=True)
		masked_input = input_data * mask

		bias_init = -masked_input.sum(dim=[0,1], keepdims=True) / num_exp
		self.bias.data = bias_init

		var_data = ( ( (input_data + bias_init)**2 ) * mask).sum(dim=[0,1], keepdims=True) / num_exp
		scaling_init = -0.5*var_data.log()
		self.scales.data = scaling_init

		out = (masked_input + self.bias) * torch.exp(self.scales)
		out_mean = (out*mask).sum(dim=[0,1]) / num_exp.squeeze()
		out_var = torch.sqrt(( ( (out - out_mean)**2 ) * mask).sum(dim=[0,1]) / num_exp)
		print("[INFO - ActNorm] New mean", out_mean)
		print("[INFO - ActNorm] New variance", out_var)


	def info(self):
		return "Activation Normalizing Flow (c_in=%i)" % (self.c_in)


class ExtActNormFlow(FlowLayer):
	"""
	Normalizes the activations over channels
	"""


	def __init__(self, c_in, net, zero_init=False, data_init=False, make_unique=False):
		super().__init__()
		self.c_in = c_in 
		self.data_init = data_init
		self.make_unique = make_unique

		self.pred_net = net
		if zero_init:
			if hasattr(self.pred_net, "initialize_zeros"):
				self.pred_net.initialize_zeros()
			elif isinstance(self.pred_net, nn.Sequential):
				self.pred_net[-1].weight.data.zero_()
				self.pred_net[-1].bias.data.zero_()


	def _run_nn(self, ext_input):
		if not self.make_unique:
			return self.pred_net(ext_input)
		else:
			orig_shape = ext_input.shape
			unique_inputs = torch.unique(ext_input)
			unique_outs = self.pred_net(unique_inputs)
			unique_inputs = unique_inputs.view(1, -1)
			ext_input = ext_input.view(-1, 1)
			indices = ((ext_input == unique_inputs).long() * torch.arange(unique_inputs.shape[1], dtype=torch.long, device=ext_input.device).unsqueeze(dim=0)).sum(dim=1)
			ext_out = unique_outs.index_select(dim=0, index=indices)
			ext_out = ext_out.reshape(orig_shape + unique_outs.shape[-1:])
			return ext_out


	def forward(self, z, ldj=None, reverse=False, ext_input=None, channel_padding_mask=None, layer_share_dict=None, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0),)
		if channel_padding_mask is None:
			channel_padding_mask = 1.0

		if ext_input is None:
			print("[!] WARNING: External input in ExtActNormFlow is None. Using default params...")
			bias = z.new_zeros(z.size(0), z.size(1), z.size(2))
			scales = bias
		else:
			nn_out = self._run_nn(ext_input)
			bias, scales = nn_out.chunk(2, dim=2)
			scales = torch.tanh(scales)
		
		if not reverse:
			z = (z + bias) * torch.exp(scales)
			ldj += (scales * channel_padding_mask).sum(dim=[1,2]) 
			if layer_share_dict is not None:
				layer_share_dict["t"] = (layer_share_dict["t"] + bias) * torch.exp(scales)
				layer_share_dict["log_s"] = layer_share_dict["log_s"] + scales
		else:
			z = z * torch.exp(-scales) - bias
			ldj += -(scales * channel_padding_mask).sum(dim=[1,2])

		assert torch.isnan(z).sum() == 0, "[!] ERROR: z contains NaN values."
		assert torch.isnan(ldj).sum() == 0, "[!] ERROR: ldj contains NaN values."
	
		return z, ldj


	def need_data_init(self):
		return self.data_init


	def data_init_forward(self, input_data, channel_padding_mask=None, **kwargs):
		if channel_padding_mask is None:
			channel_padding_mask = input_data.new_ones(input_data.shape)
		else:
			channel_padding_mask = channel_padding_mask.view(input_data.shape[:-1] + channel_padding_mask.shape[-1:])
		mask = channel_padding_mask
		num_exp = mask.sum(dim=[0,1], keepdims=True)
		masked_input = input_data

		bias_init = -masked_input.sum(dim=[0,1], keepdims=True) / num_exp

		var_data = ( ( (input_data + bias_init)**2 ) * mask).sum(dim=[0,1], keepdims=True) / num_exp
		scaling_init = -0.5*var_data.log()

		bias = torch.cat([bias_init, scaling_init], dim=-1).squeeze()

		if isinstance(self.pred_net, nn.Sequential):
			self.pred_net[-1].bias.data = bias
		else:
			self.pred_net.set_bias(bias)

		out = (masked_input + bias_init) * torch.exp(scaling_init)
		out_mean = (out*mask).sum(dim=[0,1]) / num_exp.squeeze()
		out_var = torch.sqrt(( ( (out - out_mean)**2 ) * mask).sum(dim=[0,1]) / num_exp)
		print("[INFO - External ActNorm] New mean", out_mean)
		print("[INFO - External ActNorm] New variance", out_var)


	def info(self):
		return "External Activation Normalizing Flow (c_in=%i)" % (self.c_in)



if __name__ == "__main__":
	pass