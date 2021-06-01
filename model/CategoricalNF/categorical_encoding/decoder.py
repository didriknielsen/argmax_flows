import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../")

from ..general.mutils import get_param_val
from ..networks.help_layers import LinearNet


def create_embed_layer(vocab, vocab_size, default_embed_layer_dims):
	## Creating an embedding layer either from a torchtext vocabulary or from scratch
	use_vocab_vectors = (vocab is not None and vocab.vectors is not None)
	embed_layer_dims = vocab.vectors.shape[1] if use_vocab_vectors else default_embed_layer_dims
	vocab_size = len(vocab) if use_vocab_vectors else vocab_size
	embed_layer = nn.Embedding(vocab_size, embed_layer_dims)
	if use_vocab_vectors:
		embed_layer.weight.data.copy_(vocab.vectors)
		embed_layer.weight.requires_grad = True
	return embed_layer, vocab_size


def create_decoder(num_categories, num_dims, config, **kwargs):
	num_layers = get_param_val(config, "num_layers", 1)
	hidden_size = get_param_val(config, "hidden_size", 64)

	return DecoderLinear(num_categories, 
						 embed_dim=num_dims, 
						 hidden_size=hidden_size, 
						 num_layers=num_layers,
						 **kwargs)


class DecoderLinear(nn.Module):
	"""
	A simple linear decoder with flexible number of layers. 
	"""

	def __init__(self, num_categories, embed_dim, hidden_size, num_layers, class_prior_log=None):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.layers = LinearNet(c_in=3*embed_dim, 
								c_out=num_categories,
								hidden_size=hidden_size,
								num_layers=num_layers)
		self.log_softmax = nn.LogSoftmax(dim=-1)

		if class_prior_log is not None:
			if isinstance(class_prior_log, np.ndarray):
				class_prior_log = torch.from_numpy(class_prior_log)
			self.layers.set_bias(class_prior_log)

	def forward(self, z_cont):
		z_cont = torch.cat([z_cont, F.elu(z_cont), F.elu(-z_cont)], dim=-1)
		out = self.layers(z_cont)
		logits = self.log_softmax(out)
		return logits

	def info(self):
		return "Linear model with hidden size %i and %i layers" % (self.hidden_size, self.num_layers)