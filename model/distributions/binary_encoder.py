from survae.distributions import ConditionalDistribution
from survae.transforms import Softplus
from ..transforms.utils import integer_to_base


class BinaryEncoder(ConditionalDistribution):
    '''An encoder for BinaryProductArgmaxSurjection.'''

    def __init__(self, noise_dist, dims):
        super(BinaryEncoder, self).__init__()
        self.noise_dist = noise_dist
        self.dims = dims
        self.softplus = Softplus()

    def sample_with_log_prob(self, context):
        # Example: context.shape = (B, C, H, W) with values in {0,1,...,K-1}
        # Sample z.shape = (B, C, H, W, K)

        binary = integer_to_base(context, base=2, dims=self.dims)
        sign = binary * 2 - 1

        u, log_pu = self.noise_dist.sample_with_log_prob(context=context)
        u_positive, ldj = self.softplus(u)

        log_pu_positive = log_pu - ldj
        z = u_positive * sign

        log_pz = log_pu_positive
        return z, log_pz
