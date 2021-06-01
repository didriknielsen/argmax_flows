import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../../")
from .coupling_layer import CouplingLayer


class MixtureCDFCoupling(CouplingLayer):


	def __init__(self, c_in, mask,
					   model_func,
					   block_type=None,
					   num_mixtures=10,
					   regularizer_max=-1,
					   regularizer_factor=1, **kwargs):
		"""
		Logistic mixture coupling layer as applied in Flow++.
		Parameters:
			c_in - Number of input channels
			mask - Mask to apply on the input. 1 means that the element is used as input, 0 that it is transformed
			model_func - Function for creating a model. Needs to take as input argument the number of output channels
			block_type - Name of the model. Only used for printing
			num_mixtures - Number of mixtures to apply in the layer
			regularizer_max - Mixture coupling layers apply a iterative algorithm to invert the transformations, which
							  is limited in precision. To prevent precision errors, we regularize the CDF to be between
							  10^(-regularizer_max) and 1-10^(-regularizer_max). A value of 3.5 usually works well without
							  any noticable decrease in performance. Default of -1 means no regularization.
							  This parameter should be used if sampling is important (e.g. in molecule generation)
			regularizer_factor - Factor with which to multiply the regularization loss. Commonly a value of 1 or 2 works well.
		"""
		super().__init__(c_in=c_in, mask=mask,
						 model_func=model_func,
						 block_type=block_type,
						 c_out=c_in*(2 + num_mixtures * 3),
						 **kwargs)
		self.num_mixtures = num_mixtures
		self.mixture_scaling_factor = nn.Parameter(torch.zeros(self.c_in, self.num_mixtures))
		self.regularizer_max = regularizer_max
		self.regularizer_factor = regularizer_factor


	def forward(self, z, ldj=None, reverse=False, channel_padding_mask=None, **kwargs):

		if ldj is None:
			ldj = z.new_zeros(z.size(0),)
		if channel_padding_mask is None:
			channel_padding_mask = torch.ones_like(z)

		# Mask input so that we only use the un-masked regions as input
		orig_z = z
		mask = self._prepare_mask(self.mask, z)
		z_in = z * mask

		nn_out = self.run_network(x=z_in, **kwargs)

		t, log_s, log_pi, mixt_t, mixt_log_s = MixtureCDFCoupling.get_mixt_params(nn_out, mask,
																				 num_mixtures=self.num_mixtures,
																				 scaling_factor=self.scaling_factor,
																				 mixture_scaling_factor=self.mixture_scaling_factor)
		orig_z = orig_z.double()
		z_out, ldj, reg_ldj = MixtureCDFCoupling.run_with_params(orig_z=orig_z,
																  t=t, log_s=log_s, log_pi=log_pi,
																  mixt_t=mixt_t, mixt_log_s=mixt_log_s,
																  reverse=reverse,
																  is_training=self.training,
																  reg_max=self.regularizer_max,
																  reg_factor=self.regularizer_factor,
																  mask=mask,
																  channel_padding_mask=channel_padding_mask,
																  return_reg_ldj=True)

		z_out = z_out.float()
		ldj = ldj.float()
		z_out = z_out * channel_padding_mask

		detail_out = {"ldj": ldj}
		if reg_ldj is not None:
			detail_out["regularizer_ldj"] = reg_ldj.float().sum(dim=[1,2])

		assert torch.isnan(z_out).sum() == 0 and torch.isnan(ldj).sum() == 0, "[!] ERROR: Found NaN in Mixture Coupling layer. Layer info: %s\n" % self.info() + \
				"LDJ NaN: %s, Z out NaN: %s, Z in NaN: %s, NN out NaN: %s\n" % (str(torch.isnan(ldj).sum().item()), str(torch.isnan(z_out).sum().item()), str(torch.isnan(orig_z).sum().item()), str(torch.isnan(nn_out).sum().item())) + \
				"Max/Min transition t: %s / %s\n" % (str(t.max().item()), str(t.min().item())) + \
				"Max/Min log scaling s: %s / %s\n" % (str(log_s.max().item()), str(log_s.min().item())) + \
				"Max/Min log pi: %s / %s\n" % (str(log_pi.max().item()), str(log_pi.min().item())) + \
				"Max/Min mixt t: %s / %s\n" % (str(mixt_t.max().item()), str(mixt_t.min().item())) + \
				"Max/Min mixt log s: %s / %s\n" % (str(mixt_log_s.max().item()), str(mixt_log_s.min().item())) + \
				"Mixt ldj NaN: %s\n" % (str(torch.isnan(mixt_ldj).sum().item())) + \
				"Logistic ldj NaN: %s\n" % (str(torch.isnan(logistic_ldj).sum().item()))

		return z_out, ldj, detail_out


	@staticmethod
	def run_with_params(orig_z, t, log_s, log_pi, mixt_t, mixt_log_s, reverse=False,
						reg_max=-1, reg_factor=1, mask=None, channel_padding_mask=None,
						is_training=True, return_reg_ldj=False):
		change_mask = 1-mask if mask is not None else torch.ones_like(orig_z)
		if channel_padding_mask is not None:
			change_mask = change_mask * channel_padding_mask
		reg_ldj = None
		if not reverse:
			# Calculate CDF function for given mixtures and input
			z_out = mixture_log_cdf(x=orig_z, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s).exp()

			# Regularize mixtures if wanted (only done during training as LDJ is increased)
			if reg_max > 0 and is_training:
				reg_ldj = torch.stack([safe_log(z_out), safe_log(1-z_out)], dim=-1)/np.log(10) # Change to 10 base
				reg_ldj = reg_ldj.clamp(max=-reg_max) + reg_max
				reg_ldj = reg_ldj.sum(dim=-1)
				reg_ldj = reg_ldj * change_mask
			else:
				reg_ldj = torch.zeros_like(z_out)

			# Map from [0,1] domain back to [-inf,inf] by inverse sigmoid
			z_out, mixt_ldj = inverse(z_out)
			# Output affine transformation
			z_out = (z_out + t) * log_s.exp()
			# Determine LDJ of the transformation
			logistic_ldj = mixture_log_pdf(orig_z, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s)
			# Combine all LDJs
			ldj = (change_mask * (log_s + mixt_ldj + logistic_ldj + reg_ldj * reg_factor)).sum(dim=[1,2])
		else:
			# Reverse output affine transformation
			z_out = orig_z * (-log_s).exp() - t
			# Apply sigmoid to map back to [0,1] domain
			z_out, mixt_ldj = inverse(z_out, reverse=True)
			# Clamping to prevent numerical instabilities
			z_out = z_out.clamp(1e-16, 1. - 1e-16)
			# Inverse the cummulative distribution function of mixtures. Iterative algorithm, maps back to [-inf,inf]
			z_out = mixture_inv_cdf(z_out, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s, eps=1e-10)
			# Determine LDJ of this transformation
			logistic_ldj = mixture_log_pdf(z_out, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s)
			# Combine all LDJs (note the negative sign)
			ldj = -(change_mask * (log_s + mixt_ldj + logistic_ldj)).flatten(1).sum(1)#.sum(dim=[1,2])
		if mask is not None: # Applied to ensure that the masked elements are not changed by numerical inaccuracies
			z_out = z_out * change_mask + orig_z * (1 - change_mask)
		if return_reg_ldj:
			return z_out, ldj, reg_ldj
		else:
			return z_out, ldj


	@staticmethod
	def get_mixt_params(nn_out, mask, num_mixtures, scaling_factor=None, mixture_scaling_factor=None):
		# Split network output into transformation parameters
		param_num = 2 + num_mixtures * 3
		nn_out = nn_out.reshape(nn_out.shape[:-1] + (nn_out.shape[-1]//param_num, param_num))
		t = nn_out[..., 0]
		log_s = nn_out[..., 1]
		log_pi = nn_out[..., 2:2+num_mixtures]
		mixt_t = nn_out[..., 2+num_mixtures:2+2*num_mixtures]
		mixt_log_s = nn_out[..., 2+2*num_mixtures:2+3*num_mixtures]

		# Stabilizing the scaling
		if scaling_factor is not None:
			scaling_fac = scaling_factor.exp().view(*(tuple([1 for _ in range(len(log_s.shape)-1)])+scaling_factor.shape))
			log_s = torch.tanh(log_s / scaling_fac.clamp(min=1.0)) * scaling_fac
		if mixture_scaling_factor is not None:
			mixt_fac = mixture_scaling_factor.exp().view(*(tuple([1 for _ in range(len(mixt_log_s.shape)-2)])+mixture_scaling_factor.shape))
			mixt_log_s = torch.tanh(mixt_log_s / mixt_fac.clamp(min=1.0)) * mixt_fac

		# Masking parameters
		if mask is not None:
			t = t * (1 - mask)
			log_s = log_s * (1 - mask)
			mask_ext = mask.unsqueeze(dim=-1)
			log_pi = log_pi * (1 - mask_ext) # Not strictly necessary but done for safety
			mixt_t = mixt_t * (1 - mask_ext)
			mixt_log_s = mixt_log_s * (1 - mask_ext)

		# Converting to double to prevent any numerical issues
		t = t.double()
		log_s = log_s.double()
		log_pi = log_pi.double()
		mixt_t = mixt_t.double()
		mixt_log_s = mixt_log_s.double()

		return t, log_s, log_pi, mixt_t, mixt_log_s


	def info(self):
		is_channel_mask = (self.mask.size(0) == 1)
		info_str = "Mixture CDF Coupling Layer - Input size %i" % (self.c_in)
		if self.block_type is not None:
			info_str += ", block type %s" % (self.block_type)
		info_str += ", %i mixtures" % (self.num_mixtures) + \
					", mask ratio %.2f, %s mask" % ((1-self.mask).mean().item(), "channel" if is_channel_mask else "chess")
		return info_str


"""
The following code is strongly inspired by: https://github.com/chrischute/flowplusplus
"""

def safe_log(x):
	return torch.log(x.clamp(min=1e-22))


def _log_pdf(x, mean, log_scale):
	"""Element-wise log density of the logistic distribution."""
	z = (x - mean) * torch.exp(-log_scale)
	log_p = z - log_scale - 2 * F.softplus(z)

	return log_p


def _log_cdf(x, mean, log_scale):
	"""Element-wise log CDF of the logistic distribution."""
	z = (x - mean) * torch.exp(-log_scale)
	log_p = F.logsigmoid(z)

	return log_p


def mixture_log_pdf(x, prior_logits, means, log_scales):
	"""Log PDF of a mixture of logistic distributions."""
	log_ps = F.log_softmax(prior_logits, dim=-1) \
		+ _log_pdf(x.unsqueeze(dim=-1), means, log_scales)
	log_p = torch.logsumexp(log_ps, dim=-1)

	return log_p


def mixture_log_cdf(x, prior_logits, means, log_scales):
	"""Log CDF of a mixture of logistic distributions."""
	log_ps = F.log_softmax(prior_logits, dim=-1) \
		+ _log_cdf(x.unsqueeze(dim=-1), means, log_scales)
	log_p = torch.logsumexp(log_ps, dim=-1)

	return log_p


def mixture_inv_cdf(y, prior_logits, means, log_scales,
            		eps=1e-10, max_iters=100):
	# Inverse CDF of a mixture of logisitics. Iterative algorithm.
	if y.min() <= 0 or y.max() >= 1:
		raise RuntimeError('Inverse logisitic CDF got y outside (0, 1)')

	def body(x_, lb_, ub_):
		cur_y = torch.exp(mixture_log_cdf(x_, prior_logits, means,
		                                  log_scales))
		gt = (cur_y > y).type(y.dtype)
		lt = 1 - gt
		new_x_ = gt * (x_ + lb_) / 2. + lt * (x_ + ub_) / 2.
		new_lb = gt * lb_ + lt * x_
		new_ub = gt * x_ + lt * ub_
		return new_x_, new_lb, new_ub

	x = torch.zeros_like(y)
	max_scales = torch.sum(torch.exp(log_scales), dim=-1, keepdim=True)
	lb, _ = (means - 20 * max_scales).min(dim=-1)
	ub, _ = (means + 20 * max_scales).max(dim=-1)
	diff = float('inf')

	i = 0
	while diff > eps and i < max_iters:
		new_x, lb, ub = body(x, lb, ub)
		diff = (new_x - x).abs().max()
		x = new_x
		i += 1

	return x


def inverse(x, reverse=False):
	"""Inverse logistic function."""
	if reverse:
		z = torch.sigmoid(x)
		ldj = F.softplus(x) + F.softplus(-x)
	else:
		z = -safe_log(x.reciprocal() - 1.)
		ldj = -safe_log(x) - safe_log(1. - x)

	return z, ldj


if __name__ == '__main__':
	## Example code for using Mixture-CDF coupling layer
	torch.manual_seed(42)
	batch_size, seq_len = 8, 16
	c_in, num_mixtures = 4, 10
	hidden_size = 128

	model_func = lambda c_out : nn.Sequential(
										nn.Linear(c_in, hidden_size),
										nn.ReLU(),
										nn.Linear(hidden_size, c_out)
									)
	mask = CouplingLayer.create_channel_mask(c_in)
	coupling_layer = MixtureCDFCoupling(c_in=c_in, mask=mask, model_func=model_func,
										block_type="Linear net", num_mixtures=num_mixtures)
	rand_inp = torch.randn(size=(batch_size, seq_len, c_in))

	z_forward, ldj_forward, _ = coupling_layer(z=rand_inp, reverse=False)
	z_reverse, ldj_reverse, _ = coupling_layer(z=z_forward, reverse=True)

	z_diff = (rand_inp - z_reverse).abs().max()
	LDJ_diff = (ldj_forward + ldj_reverse).abs().max()
	print("Max. reconstruction error", z_diff)
	print("Max. LDJ error", LDJ_diff)
