import os
import random
import numpy as np
np.set_printoptions(suppress=True,
                    precision=5,
                    linewidth=10000)

from backprop import BackProp, \
    sigmfun, sigmder, biased, \
    make_random_generator, xor_test


class EnhancedBackProp(BackProp):
    """
    Error back propagation algorithm for
    the learning of the weights of a multi-layered perceptron.
    Enhanced version: the learning of a predictor modulates the
    learning of local weights of the error-backprop.
    """

    def __init__(self,
                 outfun=sigmfun,
                 derfun=sigmder,
                 eta=0.01,
                 predict_eta=0.01,
                 n_units_per_layer=[2, 2, 1],
                 rng=make_random_generator()):
        """
        :param  outfun              transfer functions for units'output
        :param  derfun              derivative of the units'output function
        :param  eta                 error-back-prop learning rate
        :param  predict_eta         learning rate of the predictor
        :param  n_units_per_layer   number of units per each layer
        :param  rng                 random number generator object
        """
        super(self.__class__, self).__init__(
            outfun=outfun,
            derfun=derfun,
            eta=eta,
            n_units_per_layer=n_units_per_layer,
            rng=make_random_generator())

        self.num_hidden_units = sum(self.n_units_per_layer[1:])
        self.predict_weights = np.zeros(self.num_hidden_units)
        self.predict_eta = predict_eta
        self.predict_out = 0.0

        self.min_out = 0.0
        self.max_out = 1.0

        self.generate_predicted_weights_view()

    def generate_predicted_weights_view(self):

        self.layered_predict_weights = []
        hidden_units_per_layer = self.n_units_per_layer[1:]
        for idx_layer, n_layer in enumerate(hidden_units_per_layer):
            start = sum(hidden_units_per_layer[:idx_layer])
            end = start + n_layer
            self.layered_predict_weights.append(
                self.predict_weights[start:end])

    def step(self, input_pattern):
        """
        :param  input_pattern  the current input pattern
        """

        super(self.__class__, self).step(input_pattern)

        self.predictor_input = np.hstack([
            layer for layer in self.units[1:]])
        self.predictor_output = np.dot(self.predict_weights, self.predictor_input)

    def update_weights(self):

        self.performance = 1.0 - self.get_nrmse()
        self.prediction_error = self.performance - self.predictor_output

        self.predict_weights += self.predict_eta * \
            self.prediction_error * \
            self.predictor_input

        # print self.predict_weights.min(), self.predict_weights.max()

        delta_ws = [np.zeros(layer.shape) for layer in self.weights]
        for layer in xrange(self.n_layers - 1):
            # error plus derivative
            delta_ws[layer] = self.eta * np.outer(
                self.deltas[layer + 1],
                biased(self.units[layer]))

        for i in xrange(self.n_layers - 1):
            delta_ws[i] = ((1.0 - self.layered_predict_weights[i]) * delta_ws[i].T).T
            self.weights[i] += delta_ws[i]

    def get_nrmse(self):

        data_rng = self.max_out - self.min_out
        return np.sqrt(self.get_mse()) / data_rng


def test(predict_eta=0.0, simple=False):
    if simple == True:
        bp = BackProp(
            eta=1.0,
            n_units_per_layer=[2, 2, 1])
    else:
        bp = EnhancedBackProp(
            eta=1.0,
            predict_eta=predict_eta,
            n_units_per_layer=[2, 2, 1])
    return xor_test(bp, epochs = 10000)


if __name__ == "__main__":

    epochs = 10000
    samples = 20
    data = dict()
    data["simple"] = np.zeros([epochs, samples])
    data["enhanced_0"] = np.zeros([epochs, samples])
    data["enhanced_1"] = np.zeros([epochs, samples])

    for sample in xrange(samples):
        data["simple"][:,sample] = test(simple=True)
        data["enhanced_0"][:,sample] = test(simple=False, predict_eta=0.05)
        data["enhanced_1"][:,sample] = test(simple=False, predict_eta=0.1)

    import matplotlib.pyplot as plt

    plt.figure()

    colors= ["red", "green", "blue"]

    for color,(name, datum) in zip(colors, data.iteritems()):
        m = datum.mean(1)
        s = datum.std(1)
        plt.fill_between(np.arange(epochs), m-s, m+s,
                         facecolor=color, lw=0.01, alpha=0.5)
        plt.plot(m, color=color, lw=2)
    plt.show()
