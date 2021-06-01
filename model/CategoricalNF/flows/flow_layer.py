import torch
import torch.nn as nn 


class FlowLayer(nn.Module):

	def __init__(self):
		super().__init__()


	def forward(self, z, ldj=None, reverse=False, **kwargs):
		raise NotImplementedError


	def reverse(self, z, ldj=None, **kwargs):
		return self.forward(z, ldj, reverse=True, **kwargs)


	def need_data_init(self):
		# This function indicates whether a specific flow needs a data-dependent initialization
		# or not. For instance, activation normalization requires such a initialization
		return False


	def data_init_forward(self, input_data, **kwargs):
		# Only necessary if need_data_init is True. Contains processing of data initialization
		raise NotImplementedError


	def info(self):
		# Function to retrieve small summary/info string
		raise NotImplementedError