import torch
from survae.utils import sum_except_batch
from survae.transforms.bijections.functional.mixtures import gaussian_mixture_transform, logistic_mixture_transform, censored_logistic_mixture_transform
from survae.transforms.bijections.functional.mixtures import get_mixture_params
from .ar import ConditionalAutoregressiveBijection


class ConditionalGaussianMixtureAutoregressiveBijection(ConditionalAutoregressiveBijection):

    def __init__(self, ar_net, scheme, num_mixtures):
        super(ConditionalGaussianMixtureAutoregressiveBijection, self).__init__(ar_net=ar_net, scheme=scheme)
        self.num_mixtures = num_mixtures
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _num_params(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, params, inverse):
        assert params.shape[-1] == self._num_params()

        logit_weights, means, log_scales = get_mixture_params(params, num_mixtures=self.num_mixtures)

        x = gaussian_mixture_transform(inputs=inputs,
                                       logit_weights=logit_weights,
                                       means=means,
                                       log_scales=log_scales,
                                       eps=self.eps,
                                       max_iters=self.max_iters,
                                       inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _forward(self, x, params):
        return self._elementwise(x, params, inverse=False)

    def _element_inverse(self, z, element_params):
        return self._elementwise(z, element_params, inverse=True)


class ConditionalLogisticMixtureAutoregressiveBijection(ConditionalAutoregressiveBijection):

    def __init__(self, ar_net, scheme, num_mixtures):
        super(ConditionalLogisticMixtureAutoregressiveBijection, self).__init__(ar_net=ar_net, scheme=scheme)
        self.num_mixtures = num_mixtures
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _num_params(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, params, inverse):
        assert params.shape[-1] == self._num_params()

        logit_weights, means, log_scales = get_mixture_params(params, num_mixtures=self.num_mixtures)

        x = logistic_mixture_transform(inputs=inputs,
                                       logit_weights=logit_weights,
                                       means=means,
                                       log_scales=log_scales,
                                       eps=self.eps,
                                       max_iters=self.max_iters,
                                       inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _forward(self, x, params):
        return self._elementwise(x, params, inverse=False)

    def _element_inverse(self, z, element_params):
        return self._elementwise(z, element_params, inverse=True)


class ConditionalCensoredLogisticMixtureAutoregressiveBijection(ConditionalAutoregressiveBijection):

    def __init__(self, ar_net, scheme, num_mixtures, num_bins):
        super(ConditionalCensoredLogisticMixtureAutoregressiveBijection, self).__init__(ar_net=ar_net, scheme=scheme)
        self.num_mixtures = num_mixtures
        self.num_bins = num_bins
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _num_params(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, params, inverse):
        assert params.shape[-1] == self._num_params()

        logit_weights, means, log_scales = get_mixture_params(params, num_mixtures=self.num_mixtures)

        x = censored_logistic_mixture_transform(inputs=inputs,
                                                logit_weights=logit_weights,
                                                means=means,
                                                log_scales=log_scales,
                                                num_bins=self.num_bins,
                                                eps=self.eps,
                                                max_iters=self.max_iters,
                                                inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _forward(self, x, params):
        return self._elementwise(x, params, inverse=False)

    def _element_inverse(self, z, element_params):
        return self._elementwise(z, element_params, inverse=True)
