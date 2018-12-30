import torch
import torch.nn as nn
import math


class Gaussian(object):
    """Gaussian distribution to act as pior distribution for variational
    inference"""

    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.rho * epsilon

    def log_prob(self, input):
        return torch.sum(-math.log(math.sqrt(2 * math.pi)) - torch.log(
            self.sigma) - ((input - self.mu)**2) / (2 * self.sigma**2))
