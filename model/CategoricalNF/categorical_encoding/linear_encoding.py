import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
sys.path.append("../")

from ..general.mutils import get_param_val, one_hot
from ..flows.flow_layer import FlowLayer
from ..flows.permutation_layers import InvertibleConv
from ..flows.activation_normalization import ExtActNormFlow
from ..flows.coupling_layer import CouplingLayer
from ..flows.distributions import LogisticDistribution
from ..networks.help_layers import SimpleLinearLayer, LinearNet
from ..categorical_encoding.decoder import create_decoder, create_embed_layer

class LinearCategoricalEncoding(FlowLayer):
	"""
	Class for implementing the mixture model and linear flow encoding scheme of Categorical Normalizing Flows.
	A mixture model can be achieved by using a single activation normalization layer as "linear flow".
	Hence, this class combines both encoding schemes. 
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
	
		self.prior_distribution = LogisticDistribution(mu=0.0, sigma=1.0) # Prior distribution in encoding flows
		self.flow_layers = _create_flows(num_dims=num_dimensions, 
										 embed_dims=self.embed_layer.weight.shape[1], 
										 config=flow_config)
		# Create decoder if needed
		if self.use_decoder:
			self.decoder = create_decoder(num_categories=self.vocab_size, 
										   num_dims=self.D,
										   config=decoder_config)

		# Prior over the categories. If not given, a uniform prior is assumed
		if category_prior is None:
			category_prior = torch.zeros(self.vocab_size, dtype=torch.float32)
		else:
			assert category_prior.shape[0] == self.num_categories, "[!] ERROR: Category prior needs to be of size [%i] but is %s" % (self.num_categories, str(category_prior.shape))
			if isinstance(category_prior, np.ndarray):
				category_prior = torch.from_numpy(category_prior)
		self.register_buffer("category_prior", F.log_softmax(category_prior, dim=-1))
		

	def forward(self, z, ldj=None, reverse=False, beta=1, delta=0.0, channel_padding_mask=None, **kwargs):
		## We reshape z into [batch, 1, ...] as every categorical variable is considered to be independent.
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
			if not self.use_decoder:
				class_prior_log = torch.take(self.category_prior, z_categ.squeeze(dim=-1))
				log_point_prob = init_log_p - ldj_forward + class_prior_log
				class_prob_log = self._calculate_true_posterior(z_cont, z_categ, log_point_prob)
			else:
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

			if not self.use_decoder:
				z_out = self._posterior_sample(z_cont)
			else:
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

		return z_out, ldj, detailed_ldj


	def _flow_forward(self, z_cont, z_categ, reverse, **kwargs):
		ldj = z_cont.new_zeros(z_cont.size(0), dtype=torch.float32)
		embed_features = self.embed_layer(z_categ)
		
		for flow in (self.flow_layers if not reverse else reversed(self.flow_layers)):
			z_cont, ldj = flow(z_cont, ldj, ext_input=embed_features, reverse=reverse, **kwargs)

		return z_cont, ldj


	def _decoder_forward(self, z_cont, z_categ, **kwargs):
		## Applies the deocder on every continuous variable independently and return probability of GT class
		class_prob_log = self.decoder(z_cont)
		class_prob_log = class_prob_log.gather(dim=-1, index=z_categ.view(-1,1))
		return class_prob_log


	def _calculate_true_posterior(self, z_cont, z_categ, log_point_prob, **kwargs):
		## Run backward pass of *all* class-conditional flows
		z_back_in = z_cont.expand(-1, self.num_categories, -1).reshape(-1, 1, z_cont.size(2))
		sample_categ = torch.arange(self.num_categories, dtype=torch.long).to(z_cont.device)
		sample_categ = sample_categ[None,:].expand(z_categ.size(0), -1).reshape(-1, 1)

		z_back, ldj_backward = self._flow_forward(z_back_in, sample_categ, reverse=True, **kwargs)
		back_log_p = self.prior_distribution.log_prob(z_back).sum(dim=[1,2])
		
		## Calculate the denominator (sum of probabilities of all classes)
		flow_log_prob = back_log_p + ldj_backward
		log_prob_denominator = flow_log_prob.view(z_cont.size(0), self.num_categories) + self.category_prior[None,:]
		# Replace log_prob of original class with forward probability
		# This improves stability and prevents the model to exploit numerical errors during inverting the flows
		orig_class_mask = one_hot(z_categ.squeeze(), num_classes=log_prob_denominator.size(1))
		log_prob_denominator = log_prob_denominator * (1 - orig_class_mask) + log_point_prob.unsqueeze(dim=-1) * orig_class_mask
		# Denominator is the sum of probability -> turn log to exp, and back to log
		log_denominator = torch.logsumexp(log_prob_denominator, dim=-1)
		
		## Combine nominator and denominator for final prob log
		class_prob_log = (log_point_prob - log_denominator)
		return class_prob_log


	def _decoder_sample(self, z_cont, **kwargs):
		## Sampling from decoder by taking the argmax.
		# We could also sample from the probabilities, however experienced that the argmax gives more stable results.
		# Presumably because the decoder has also seen values sampled from the encoding distributions and not anywhere besides that.
		return self.decoder(z_cont).argmax(dim=-1)


	def _posterior_sample(self, z_cont, **kwargs):
		## Run backward pass of *all* class-conditional flows
		z_back_in = z_cont.expand(-1, self.num_categories, -1).reshape(-1, 1, z_cont.size(2))
		sample_categ = torch.arange(self.num_categories, dtype=torch.long).to(z_cont.device)
		sample_categ = sample_categ[None,:].expand(z_cont.size(0), -1).reshape(-1, 1)

		z_back, ldj_backward = self._flow_forward(z_back_in, sample_categ, reverse=True, **kwargs)
		back_log_p = self.prior_distribution.log_prob(z_back).sum(dim=[1,2])
		
		## Calculate the log probability for each class
		flow_log_prob = back_log_p + ldj_backward
		log_prob_denominator = flow_log_prob.view(z_cont.size(0), self.num_categories) + self.category_prior[None,:]
		return log_prob_denominator.argmax(dim=-1)


	def info(self):
		s = ""
		if len(self.flow_layers) > 1:
			s += "Linear Encodings of categories, with %i dimensions and %i flows.\n" % (self.D, len(self.flow_layers))
		else:
			s += "Mixture model encoding of categories with %i dimensions\n" % (self.D)
		s += "-> Prior distribution: %s\n" % self.prior_distribution.info()
		if self.use_decoder:
			s += "-> Decoder network: %s\n" % self.decoder.info()
		s += "\n".join(["-> [%i] " % (flow_index+1) + flow.info() for flow_index, flow in enumerate(self.flow_layers)])
		return s




def _create_flows(num_dims, embed_dims, config):
	num_flows = get_param_val(config, "num_flows", 0)
	num_hidden_layers = get_param_val(config, "hidden_layers", 2)
	hidden_size = get_param_val(config, "hidden_size", 256)
	
	# We apply a linear net in the coupling layers for linear flows
	block_type_name = "LinearNet"
	block_fun_coup = lambda c_out : LinearNet(c_in=num_dims,
											  c_out=c_out,
											  num_layers=num_hidden_layers,
											  hidden_size=hidden_size,
											  ext_input_dims=embed_dims)

	# For the activation normalization, we map an embedding to scaling and bias with a single layer
	block_fun_actn = lambda : SimpleLinearLayer(c_in=embed_dims, c_out=2*num_dims, data_init=True)
	
	permut_layer = lambda flow_index : InvertibleConv(c_in=num_dims)
	actnorm_layer = lambda flow_index : ExtActNormFlow(c_in=num_dims, 
													   net=block_fun_actn())
	# We do not use mixture coupling layers here aas we need the inverse to be differentiable as well
	coupling_layer = lambda flow_index : CouplingLayer(c_in=num_dims, 
													   mask=CouplingLayer.create_channel_mask(c_in=num_dims), 
													   block_type=block_type_name,
													   model_func=block_fun_coup)

	flow_layers = []
	if num_flows == 0 or num_dims == 1: # Num_flows == 0 => mixture model, num_dims == 1 => coupling layers have no effect
		flow_layers += [actnorm_layer(flow_index=0)]
	else:
		for flow_index in range(num_flows):
			flow_layers += [
				actnorm_layer(flow_index), 
				permut_layer(flow_index),
				coupling_layer(flow_index)
			]

	return nn.ModuleList(flow_layers)




if __name__ == '__main__':
	## Example for using linear encoding
	torch.manual_seed(42)
	np.random.seed(42)

	batch_size, seq_len = 3, 6
	vocab_size, D = 4, 3
	flow_config = {
		"num_flows": 0,
		"num_hidden_layers": 1,
		"hidden_size": 128
	}

	categ_encod = LinearCategoricalEncoding(num_dimensions=D, flow_config=flow_config, vocab_size=vocab_size)
	print(categ_encod.info())
	rand_inp = torch.randint(high=vocab_size, size=(batch_size, seq_len), dtype=torch.long)
	z_out, ldj, detail_ldj = categ_encod(rand_inp)
	print("Z out", z_out)
	print("Detail ldj", detail_ldj)