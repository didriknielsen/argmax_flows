import torch
import torch.nn as nn
import sys
sys.path.append("../")

from ..general.mutils import get_param_val
from ..flows.flow_layer import FlowLayer
from ..flows.sigmoid_layer import SigmoidFlow
from ..flows.activation_normalization import ActNormFlow
from ..flows.coupling_layer import CouplingLayer
from ..categorical_encoding.decoder import create_embed_layer


class VariationalDequantization(FlowLayer):
	"""
	Flow layer to encode discrete variables using variational dequantization.
	"""


	def __init__(self, flow_config, 
					   vocab=None, vocab_size=-1, 
					   default_embed_layer_dims=128,
					   **kwargs):
		super().__init__()
		self.embed_layer, self.vocab_size = create_embed_layer(vocab, vocab_size, default_embed_layer_dims)
		self.flow_layers = _create_flows(flow_config, self.embed_layer.weight.shape[1])
		self.sigmoid_flow = SigmoidFlow(reverse=True)


	def forward(self, z, ldj=None, reverse=False, **kwargs):
		batch_size, seq_length = z.size(0), z.size(1)

		if ldj is None:
			ldj = z.new_zeros(z.size(0), dtype=torch.float32)
		
		if not reverse:
			# Sample from noise distribution, modeled by the normalizing flow
			rand_inp = torch.rand_like(z, dtype=torch.float32).unsqueeze(dim=-1) 	# Output range [0,1]
			rand_inp, ldj = self.sigmoid_flow(rand_inp, ldj=ldj, reverse=False) 	# Output range [-inf,inf]
			rand_inp, ldj = self._flow_forward(rand_inp, z, ldj, **kwargs) 			# Output range [-inf,inf]
			rand_inp, ldj = self.sigmoid_flow(rand_inp, ldj=ldj, reverse=True) 		# Output range [0,1]
			# Checking that noise is indeed in the range [0,1]. Any value outside indicates a numerical issue in the dequantization flow
			assert (rand_inp<0.0).sum() == 0 and (rand_inp>1.0).sum() == 0, "ERROR: Variational Dequantization output is out of bounds.\n" + \
					str(torch.where(rand_inp<0.0)) + "\n" + \
					str(torch.where(rand_inp>1.0))
			# Adding the noise to the discrete values
			z_out = z.to(torch.float32).unsqueeze(dim=-1) + rand_inp
			assert torch.isnan(z_out).sum() == 0, "ERROR: Found NaN values in variational dequantization.\n" + \
					"NaN z_out: " + str(torch.isnan(z_out).sum().item()) + "\n" + \
					"NaN rand_inp: " + str(torch.isnan(rand_inp).sum().item()) + "\n" + \
					"NaN ldj: " + str(torch.isnan(ldj).sum().item())
		else:
			# Inverting the flow is done by finding the next whole integer for each continuous value
			z_out = torch.floor(z).clamp(min=0, max=self.vocab_size-1)
			z_out = z_out.long().squeeze(dim=-1)

		return z_out, ldj


	def _flow_forward(self, rand_inp, z, ldj, **kwargs):
		# Adding discrete values to flow transformation input by an embedding layer 
		embed_features = self.embed_layer(z)
		for flow in self.flow_layers:
			rand_inp, ldj = flow(rand_inp, ldj, ext_input=embed_features, reverse=False, **kwargs)
		return rand_inp, ldj


	def info(self):
		s = "Variational Dequantization with %i flows.\n" % (len(self.flow_layers))
		s += "\n".join(["-> [%i] " % (flow_index+1) + flow.info() for flow_index, flow in enumerate(self.flow_layers)])
		return s


def _create_flows(config, embed_dims):
	num_flows = get_param_val(config, "num_flows", 4)
	model_func = get_param_val(config, "model_func", allow_default=False)
	block_type = get_param_val(config, "block_type", None)

	def _create_block(flow_index):
		# For variational dequantization we apply a combination of activation normalization and coupling layers.
		# Invertible convolutions are not useful here as our dimensionality is 1 anyways 
		mask = CouplingLayer.create_chess_mask()
		if flow_index % 2 == 0:
			mask = 1 - mask
		return [
			ActNormFlow(c_in=1, data_init=False),
			CouplingLayer(c_in=1, 
						  mask=mask, 
						  model_func=model_func,
						  block_type=block_type)
		]

	flow_layers = []
	for flow_index in range(num_flows):
		flow_layers += _create_block(flow_index)

	return nn.ModuleList(flow_layers)


if __name__ == '__main__':
	## Example code for using variational dequantization
	torch.manual_seed(42)
	batch_size, seq_len = 3, 6
	vocab_size = 4
	hidden_size, embed_layer_dims = 128, 128

	class ExampleNetwork(nn.Module):
		def __init__(self, c_out):
			super().__init__()
			self.inp_layer = nn.Linear(1, hidden_size)
			self.main_net = nn.Sequential(
				nn.Linear(hidden_size + embed_layer_dims, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, c_out)
			)

		def forward(self, x, ext_input, **kwargs):
			inp = self.inp_layer(x)
			out = self.main_net(torch.cat([inp, ext_input], dim=-1))
			return out

	model_func = lambda c_out : ExampleNetwork(c_out)

	flow_config = {
		"num_flows": 2,
		"model_func": model_func,
		"block_type": "Linear"
	}
	vardeq_flow = VariationalDequantization(vocab_size=vocab_size,
											flow_config=flow_config,
											embed_layer_dims=embed_layer_dims)

	z = torch.randint(high=vocab_size, size=(batch_size, seq_len), dtype=torch.long)
	z_cont, _ = vardeq_flow(z, reverse=False)
	z_rec, _ = vardeq_flow(z_cont, reverse=True)

	print("-"*90)
	print(vardeq_flow.info())
	print("-"*90)
	print("Z\n", z)
	print("Z reconstructed\n", z_rec)
	print("Z continuous\n", z_cont)
	assert (z_rec == z).all()
