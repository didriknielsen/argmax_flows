import torch
import torch.nn as nn
import numpy as np 


class FlowModel(nn.Module):


	def __init__(self, layers=None, name="Flow model"):
		super().__init__()

		self.flow_layers = nn.ModuleList()
		self.name = name

		if layers is not None:
			self.add_layers(layers)


	def add_layers(self, layers):
		for l in layers:
			self.flow_layers.append(l)
		self.print_overview()


	def forward(self, z, ldj=None, reverse=False, get_ldj_per_layer=False, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0), dtype=torch.float32)

		ldj_per_layer = []
		for layer_index, layer in (enumerate(self.flow_layers) if not reverse else reversed(list(enumerate(self.flow_layers)))):
			
			layer_res = layer(z, reverse=reverse, get_ldj_per_layer=get_ldj_per_layer, **kwargs)

			if len(layer_res) == 2:
				z, layer_ldj = layer_res 
				detailed_layer_ldj = layer_ldj
			elif len(layer_res) == 3:
				z, layer_ldj, detailed_layer_ldj = layer_res
			else:
				print("[!] ERROR: Got more return values than expected: %i" % (len(layer_res)))

			assert torch.isnan(z).sum() == 0, "[!] ERROR: Found NaN latent values. Layer (%i):\n%s" % (layer_index + 1, layer.info())
			
			ldj = ldj + layer_ldj
			if isinstance(detailed_layer_ldj, list):
				ldj_per_layer += detailed_layer_ldj
			else:
				ldj_per_layer.append(detailed_layer_ldj)

		if get_ldj_per_layer:
			return z, ldj, ldj_per_layer
		else:
			return z, ldj


	def reverse(self, z):
		return self.forward(z, reverse)


	def test_reversibility(self, z, **kwargs):
		test_failed = False
		for layer_index, layer in enumerate(self.flow_layers):
			z_layer, ldj_layer = layer(z, reverse=False, **kwargs)
			z_reconst, ldj_reconst = layer(z_layer, reverse=True, **kwargs)

			ldj_diff = (ldj_layer + ldj_reconst).abs().sum()
			z_diff = (z_layer - z_reconst).abs().sum()

			if z_diff != 0 or ldj_diff != 0:
				print("-"*100)
				print("[!] WARNING: Reversibility check failed for layer index %i" % layer_index)
				print(layer.info())
				print("-"*100)
				test_failed = True

		print("+"*100)
		print("Reversibility test %s (tested %i layers)" % ("failed" if test_failed else "succeeded", len(self.flow_layers)))
		print("+"*100)


	def get_inner_activations(self, z, reverse=False, return_names=False, **kwargs):
		out_per_layer = [z.detach()]
		layer_names = []
		for layer_index, layer in enumerate((self.flow_layers if not reverse else reversed(self.flow_layers))):
			
			z = layer(z, reverse=reverse, **kwargs)[0]
			out_per_layer.append(z.detach())
			layer_names.append(layer.__class__.__name__)

		if not return_names:
			return out_per_layer
		else:
			return out_per_layer, return_names


	def initialize_data_dependent(self, batch_list):
		# Batch list needs to consist of tuples: (z, kwargs)
		with torch.no_grad():
			for layer_index, layer in enumerate(self.flow_layers):
				print("Processing layer %i..." % (layer_index+1), end="\r")
				batch_list = FlowModel.run_data_init_layer(batch_list, layer)


	@staticmethod
	def run_data_init_layer(batch_list, layer):
		if layer.need_data_init():
			stacked_kwargs = {key: [b[1][key] for b in batch_list] for key in batch_list[0][1].keys()}
			for key in stacked_kwargs.keys():
				if isinstance(stacked_kwargs[key][0], torch.Tensor):
					stacked_kwargs[key] = torch.cat(stacked_kwargs[key], dim=0)
				else:
					stacked_kwargs[key] = stacked_kwargs[key][0]
			if not (isinstance(batch_list[0][0], tuple) or isinstance(batch_list[0][0], list)):
				input_data = torch.cat([z for z, _ in batch_list], dim=0)
				layer.data_init_forward(input_data, **stacked_kwargs)
			else:
				input_data = [torch.cat([z[i] for z, _ in batch_list], dim=0) for i in range(len(batch_list[0][0]))]
				layer.data_init_forward(*input_data, **stacked_kwargs)
		out_list = []
		for z, kwargs in batch_list:
			if isinstance(z, tuple) or isinstance(z, list):
				z = layer(*z, reverse=False, **kwargs)
				out_list.append([e.detach() for e in z[:-1] if isinstance(e, torch.Tensor)])
				if len(z) == 4 and isinstance(z[-1], dict):
					kwargs.update(z[-1])
					out_list[-1] = out_list[-1][:-1]
			else:
				z = layer(z, reverse=False, **kwargs)[0]
				out_list.append(z.detach())
		batch_list = [(out_list[i], batch_list[i][1]) for i in range(len(batch_list))]
		return batch_list


	def need_data_init(self):
		return any([flow.need_data_init() for flow in self.flow_layers])


	def print_overview(self):
		# Retrieve layer descriptions for all flows
		layer_descp = list()
		for layer_index, layer in enumerate(self.flow_layers):
			layer_descp.append("(%2i) %s" % (layer_index+1, layer.info()))
		num_tokens = max([20] + [len(s) for s in "\n".join(layer_descp).split("\n")])
		# Print out info in a nicer format
		print("="*num_tokens)
		print("%s with %i flows" % (self.name, len(self.flow_layers)))
		print("-"*num_tokens)
		print("\n".join(layer_descp))
		print("="*num_tokens)