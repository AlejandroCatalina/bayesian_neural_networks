import numpy as np
import numpy.random as npr


def build_toy_dataset(n_data=40, noise_std=0.1):
    D = 1
    rs = npr.RandomState(0)
    inputs = np.concatenate(
        [np.linspace(0, 2, num=n_data / 2),
         np.linspace(6, 8, num=n_data / 2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 4.0
    inputs = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets
