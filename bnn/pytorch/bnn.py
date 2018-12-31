import torch
import torch.nn as nn
from torch.functional import F

from distributions import Gaussian


class BLinear(nn.Module):
    """Bayesian Linear layer, default prior is Gaussian for weights and bias"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(1e-1, 2))
        # variational posterior for the weights
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias parameters
        self.bias_mu = nn.Parameter(
            torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(
            torch.Tensor(out_features).uniform_(1e-1, 2))
        # variational posterior for the bias
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # Prior distributions
        self.weight_prior = Gaussian(torch.Tensor([0.]), torch.Tensor([1.]))
        self.bias_prior = Gaussian(torch.Tensor([0.]), torch.Tensor([1.]))

        # initialize log_prior and log_posterior as 0
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        # 1. Sample weights and bias from variational posterior
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        # 2. Update log_prior and log_posterior according to current approximation
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(
                weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(
                weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        # 3. Do a forward pass through the layer
        return F.linear(input, weight, bias)


class BNet(nn.Module):
    def __init__(self,
                 input_size,
                 batch_size,
                 output_size,
                 n_training,
                 loss_function=F.nll_loss):
        super().__init__()

        self.l1 = BLinear(input_size, 10)
        self.l2 = BLinear(10, 10)
        self.l3 = BLinear(10, output_size)

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_training = n_training
        self.loss_function = loss_function
        self.noise = torch.fill_(torch.zeros(batch_size), 1e-3)

    def forward(self, x, sample=False):
        x = x.view(-1, self.input_size)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = self.l3(x, sample)
        return x

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l2.log_variational_posterior

    def elbo(self, input, target, samples=20):
        outputs = torch.zeros(samples, self.batch_size, self.output_size)
        log_priors = torch.zeros(samples)
        log_variational_posteriors = torch.zeros(samples)

        # draw n_samples from the posterior (run n_samples forward passes)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()

        log_prior = log_priors.sum()
        log_variational_posterior = log_variational_posteriors.sum()

        y_dist = Gaussian(outputs.mean(0), self.noise)
        # negative_log_likelihood = self.loss_function(
        #     outputs.mean(0), target, reduction='sum')
        negative_log_likelihood = -y_dist.log_prob(target) / len(input)

        # loss = nll + kl
        loss = negative_log_likelihood
        kl = (log_variational_posterior - log_prior) / self.n_training
        loss += kl
        print(negative_log_likelihood, kl)
        return loss, log_prior, log_variational_posterior, negative_log_likelihood
