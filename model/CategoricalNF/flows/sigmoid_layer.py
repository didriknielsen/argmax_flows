import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append("../../")

from layers.flows.flow_layer import FlowLayer



class SigmoidFlow(FlowLayer):
	"""
	Applies a sigmoid on an output
	"""


	def __init__(self, reverse=False):
		super().__init__()
		self.sigmoid = nn.Sigmoid()
		self.reverse_layer = reverse


	def forward(self, z, ldj=None, reverse=False, sum_ldj=True, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0),)

		alpha = 1e-5
		reverse = (self.reverse_layer != reverse) # XOR over reverse parameters

		if not reverse:
			layer_ldj = -z - 2 * F.softplus(-z)
			z = torch.sigmoid(z)
		else:
			z = z*(1-alpha) + alpha*0.5 # Remove boundaries of 0 and 1 (which would result in minus infinity and inifinity)
			layer_ldj = (-torch.log(z) - torch.log(1-z) + math.log(1 - alpha))
			z = torch.log(z) - torch.log(1-z)

		assert torch.isnan(z).sum() == 0, "[!] ERROR: z contains NaN values."
		assert torch.isnan(layer_ldj).sum() == 0, "[!] ERROR: ldj contains NaN values."

		if sum_ldj:
			ldj = ldj + layer_ldj.view(z.size(0), -1).sum(dim=1)
		else:
			ldj = layer_ldj

		return z, ldj


	def info(self):
		return "Sigmoid Flow"