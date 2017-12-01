import os
import random
import numpy as np
np.set_printoptions(suppress=True,
                    precision=5,
                    linewidth=10000)

# ----------------------------------------------------------------------------
# utils ----------------------------------------------------------------------

# Add a bias unit to the input


def biased(x):
    return np.hstack([1, x])

# sigmoid function
# t   float temperature


def sigmfun(x, t=1.0):
    return 1.0 / (1.0 + np.exp(-x / t))

# sigmoid derivative


def sigmder(y):
    return y * (1 - y)


def relu(x):
    return np.maximum(0.0, x)


def reluder(x):
    return (x > 0.0) * 1.0

# create a numpy random number generator


def make_random_generator(SEED=None):
    if SEED is None:
        SEED = np.fromstring(os.urandom(4), dtype=np.uint32)[0]
    return np.random.RandomState(SEED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


class BackProp(object):
    """
    Error back propagation algorithm for
    the learning of the weights of a multi-layered perceptron.
    """

    def __init__(self,
                 outfun=sigmfun,
                 derfun=sigmder,
                 eta=0.01,
                 n_units_per_layer=[2, 2, 1],
                 rng=make_random_generator()):
        """
        :param  outfun              transfer functions for units'output
        :param  derfun              derivative of the units'output function
        :param  eta                 learning rate
        :param  n_units_per_layer   number of units per each layer
        :param  rng                 random number generator object
        """

        self.rng = rng

        # function pointers

        # transfer functions for units'output
        self.outfun = outfun

        # derivative of the units'output function
        self.derfun = derfun

        # Constants

        # Learning rate
        self.eta = eta

        # number of layer units
        self.n_units_per_layer = n_units_per_layer

        # number of layers
        self.n_layers = len(self.n_units_per_layer)

        # Variables

        # Initialize units
        self.units = []
        for n in self.n_units_per_layer:
            self.units.append(np.zeros(n))

        # Initialize deltas
        self.deltas = []
        for n in self.n_units_per_layer:
            self.deltas.append(np.zeros(n))

        # Initialize weights
        self.weights = []
        for n_out, n_inp in zip(self.n_units_per_layer[1:],
                                self.n_units_per_layer[:-1]):
            self.weights.append(self.rng.randn(n_out, n_inp + 1))

    def step(self, input_pattern):
        """
        :param  input_pattern  the current input pattern
        """
        # spreading

        # linear function
        self.units[0] = input_pattern.copy()

        # iterate deep layers
        for layer in xrange(self.n_layers - 1):
            # Bias-plus-input vector
            input_pattern = np.dot(self.weights[layer],
                                   biased(self.units[layer]))
            # nonlinear function
            self.units[layer + 1] = self.outfun(input_pattern)

    def error_backprop(self, target):
        """
        :param  target  the desired output for the current input
        """

        # the initial outer-error is the real one
        #     (expected vs real outputs)
        self.error = target - self.units[-1]

        self.deltas[-1] = self.error * self.derfun(self.units[-1])
        # iterate layers backward
        for layer in xrange(self.n_layers - 1, 1, -1):
            # for each layer evaluate the back-propagation error
            w_T = (self.weights[layer - 1][:, 1:]).T
            self.deltas[layer - 1] = \
                np.dot(w_T, self.deltas[layer]) * \
                self.derfun(self.units[layer - 1])

    def update_weights(self):

        for layer in xrange(self.n_layers - 1):
            # error plus derivative
            self.weights[layer] += self.eta * np.outer(
                self.deltas[layer + 1],
                biased(self.units[layer]))

    def learn(self, target):
        """
        :param  target  the desired output for the current input
        """

        self.error_backprop(target)
        self.update_weights()

    def get_mse(self):
        return np.mean(self.error**2)


