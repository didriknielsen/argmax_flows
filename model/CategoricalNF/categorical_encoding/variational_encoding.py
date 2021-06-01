import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from survae.transforms import Transform, Bijection

sys.path.append("../")

from ..general.mutils import get_param_val, one_hot
from ..flows.flow_layer import FlowLayer
from ..flows.permutation_layers import InvertibleConv
from ..flows.activation_normalization import ExtActNormFlow
from ..flows.coupling_layer import CouplingLayer
from ..flows.mixture_cdf_layer import MixtureCDFCoupling
from ..flows.distributions import LogisticDistribution, GaussianDistribution
from ..networks.help_layers import SimpleLinearLayer
from ..categorical_encoding.decoder import create_decoder, create_embed_layer

class VariationalCategoricalEncoding(Bijection):
	"""
	Class for implementing the variational encoding scheme of Categorical Normalizing Flows.
	"""

	def __init__(self, num_dimensions, flow_config,
					   dataset_class=None,
					   vocab=None, vocab_size=-1,
					   use_decoder=False, decoder_config=None,
					   default_embed_layer_dims=64,
					   category_prior=None,
					   **kwargs):
		super().__init__()
		self.use_decoder = use_decoder
		self.dataset_class = dataset_class
		self.D = num_dimensions

		self.embed_layer, self.vocab_size = create_embed_layer(vocab, vocab_size, default_embed_layer_dims)
		self.num_categories = self.vocab_size

		# self.prior_distribution = GaussianDistribution(mu=0., sigma=1.)
		self.prior_distribution = LogisticDistribution(mu=0.0, sigma=1.0) # Prior distribution in encoding flows
		self.flow_layers = _create_flows(num_dims=num_dimensions,
										 embed_dims=self.embed_layer.weight.shape[1],
										 config=flow_config)
		# Create decoder if needed
		self.decoder = create_decoder(num_categories=self.vocab_size,
									  num_dims=self.D,
									  config=decoder_config)


	def forward(self, z, reverse=False, beta=1, delta=0.0, channel_padding_mask=None, **kwargs):
		## We reshape z into [batch, 1, ...] as every categorical variable is considered to be independent.
		ldj = None
		z = z.permute(0, 2, 1).squeeze()
		batch_size, seq_length = z.size(0), z.size(1)
		z = z.reshape((batch_size * seq_length, 1) + z.shape[2:])
		if channel_padding_mask is not None:
			channel_padding_mask = channel_padding_mask.reshape(batch_size * seq_length, 1, -1)
		else:
			channel_padding_mask = z.new_ones((batch_size * seq_length, 1, 1), dtype=torch.float32)

		ldj_loc = z.new_zeros(z.size(0), dtype=torch.float32)
		detailed_ldj = {}

		if not reverse:
			# z is of shape [Batch, SeqLength]
			z_categ = z # Renaming here for better readability (what is discrete and what is continuous)

			## 1.) Forward pass of current token flow
			z_cont = self.prior_distribution.sample(shape=(batch_size * seq_length, 1, self.D)).to(z_categ.device)
			init_log_p = self.prior_distribution.log_prob(z_cont).sum(dim=[1,2])
			z_cont, ldj_forward = self._flow_forward(z_cont, z_categ, reverse=False)

			## 2.) Approach-specific calculation of the posterior
			class_prob_log = self._decoder_forward(z_cont, z_categ)

			## 3.) Calculate final LDJ
			ldj_loc = (beta * class_prob_log - (init_log_p - ldj_forward))
			ldj_loc = ldj_loc * channel_padding_mask.squeeze()
			z_cont = z_cont * channel_padding_mask
			z_out = z_cont

			## 4.) Statistics for debugging/monotoring
			if self.training:
				with torch.no_grad():
					z_min = z_out.min()
					z_max = z_out.max()
					z_std = z_out.view(-1, z_out.shape[-1]).std(0).mean()
					channel_padding_mask = channel_padding_mask.squeeze()
					detailed_ldj = {"avg_token_prob": (class_prob_log.exp() * channel_padding_mask).sum()/channel_padding_mask.sum(),
									"avg_token_bpd": -(class_prob_log * channel_padding_mask).sum()/channel_padding_mask.sum() * np.log2(np.exp(1)),
									"z_min": z_min,
									"z_max": z_max,
									"z_std": z_std}
					detailed_ldj = {key: val.detach() for key, val in detailed_ldj.items()}

		else:
			# z is of shape [Batch * seq_len, 1, D]
			assert z.size(-1) == self.D, "[!] ERROR in categorical decoding: Input must have %i latent dimensions but got %i" % (self.D, z.shape[-1])

			class_prior_log = self.category_prior[None,None,:]
			z_cont = z
			z_out = self._decoder_sample(z_cont)

		# Reshape output back to original shape
		if not reverse:
			z_out = z_out.reshape(batch_size, seq_length, -1)
		else:
			z_out = z_out.reshape(batch_size, seq_length)

		ldj_loc = ldj_loc.reshape(batch_size, seq_length).sum(dim=-1)

		# Add LDJ
		if ldj is not None:
			ldj = ldj + ldj_loc
		else:
			ldj = ldj_loc

		z_out = z_out.permute(0, 2, 1).unsqueeze(1)

		return z_out, ldj  #, detailed_ldj


	def _flow_forward(self, z_cont, z_categ, reverse, **kwargs):
		ldj = z_cont.new_zeros(z_cont.size(0), dtype=torch.float32)
		embed_features = self.embed_layer(z_categ)

		for flow in (self.flow_layers if not reverse else reversed(self.flow_layers)):
			z_cont, ldj = flow(z_cont, ldj, ext_input=embed_features, reverse=reverse, **kwargs)

		return z_cont, ldj


	def _decoder_forward(self, z_cont, z_categ, **kwargs):
		## Applies the deocder on every continuous variable independently and return probability of GT class
		class_prob_log = self.decoder(z_cont)

		class_prob_log = class_prob_log.gather(dim=-1, index=z_categ.view(-1,1,1))
		class_prob_log = class_prob_log.squeeze()
		return class_prob_log


	def _decoder_sample(self, z_cont, **kwargs):
		## Sampling from decoder by taking the argmax.
		# We could also sample from the probabilities, however experienced that the argmax gives more stable results.
		# Presumably because the decoder has also seen values sampled from the encoding distributions and not anywhere besides that.
		return self.decoder(z_cont).argmax(dim=-1)


	def info(self):
		s = "Variational Encodings of categories, with %i dimensions and %i flows.\n" % (self.D, len(self.flow_layers))
		s += "-> Decoder network: %s\n" % self.decoder.info()
		s += "\n".join(["-> [%i] " % (flow_index+1) + flow.info() for flow_index, flow in enumerate(self.flow_layers)])
		return s




def _create_flows(num_dims, embed_dims, config):
	num_flows = get_param_val(config, "num_flows", 0)
	model_func = get_param_val(config, "model_func", allow_default=False)
	block_type = get_param_val(config, "block_type", None)
	num_mixtures = get_param_val(config, "num_mixtures", 8)

	# For the activation normalization, we map an embedding to scaling and bias with a single layer
	block_fun_actn = lambda : SimpleLinearLayer(c_in=embed_dims, c_out=2*num_dims, data_init=True)

	permut_layer = lambda flow_index : InvertibleConv(c_in=num_dims)
	actnorm_layer = lambda flow_index : ExtActNormFlow(c_in=num_dims,
													   net=block_fun_actn())

	if num_dims > 1:
		mask = CouplingLayer.create_channel_mask(c_in=num_dims)
		mask_func = lambda _ : mask
	else:
		mask = CouplingLayer.create_chess_mask()
		mask_func = lambda flow_index : mask if flow_index%2 == 0 else 1-mask

	coupling_layer = lambda flow_index : MixtureCDFCoupling(c_in=num_dims,
															mask=mask_func(flow_index),
															block_type=block_type,
															model_func=model_func,
															num_mixtures=num_mixtures)

	flow_layers = []
	if num_flows == 0: # Num_flows == 0 => mixture model
		flow_layers += [actnorm_layer(flow_index=0)]
	else:
		for flow_index in range(num_flows):
			flow_layers += [
				actnorm_layer(flow_index),
				permut_layer(flow_index),
				coupling_layer(flow_index)
			]

	return nn.ModuleList(flow_layers)
