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
        for layer in xrange(self.n_layers - 1, 0, -1):
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

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# test


def xor_test(bp, epochs = 5000):
    """
    Classical xor example

    :param  bp              a BackProp object
    :param  epochs          number of epochs
    """

    # define the dataset (columns are input1, input2, target)
    data = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0]])

    # we build a vector of dataset
    #     indices to shuffle
    idcs = range(4)

    # init error storage
    error_means = np.zeros(epochs)

    # learning loop
    for t in xrange(epochs):
        # give each pattern of the dataset
        #      in a random order
        random.shuffle(idcs)
        # storage for epoch errors
        errors = np.zeros(4)
        for p in idcs:
            # backprop
            bp.step(data[p, :2])
            bp.learn(data[p, 2])
            errors[p] = bp.get_mse()
        # mean of error in the epopch
        error_means[t] = errors.mean()

        print "% 8d % 10.6f" % (t, error_means[t])

    return error_means

def backprop_xor_test():
    """
    Classical xor example (using BackProp)
    """
    # init the backprop object
    bp = BackProp(
        outfun=sigmfun, derfun=sigmder,
        n_units_per_layer=[2, 2, 1],
        eta=1)

    xor_test(bp)

def mnist_test(bp):
    """
    Use backprop to teach an autoencoder
        over the mnist dataset

    :param  bp a BackProp object
    """

    # mnist

    # import the mnist class
    from mnist import MNIST

    # init with the 'data' dir
    mndata = MNIST('./data')

    # Load data
    mndata.load_training()
    mndata.load_testing()

    # The number of pixels per side of all images
    img_side = 28

    # Each input is a raw vector.
    # The number of units of the network
    # corresponds to the number of input elements
    n_mnist_pixels = img_side * img_side

    # lengths of datatasets patterns
    n_train = len(mndata.train_images)
    n_test = len(mndata.test_images)

    # number of tests
    tests = 8

    # number of epochs
    n_epochs = 300

    # the display object allows dynamic plots on jupyter
    from IPython import display

    # Init plots

    import matplotlib.pyplot as plt
    plt.ioff()

    fig1 = plt.figure(figsize=(12, 24))

    # init plots for target mnist digits
    (ax1, img1, ax2, img2, ax3, img3, ax4, img4) = (dict() for x in xrange(8))
    for img_n in xrange(tests):

        # train plots

        # init plots for target mnist digits
        ax1[img_n] = fig1.add_subplot(tests, 4, 1 + 4 * img_n)
        img1[img_n] = ax1[img_n].imshow(np.zeros([img_side, img_side]),
                                        vmin=0, vmax=1)

        # init plots for reproduced mnist digit
        ax2[img_n] = fig1.add_subplot(tests, 4, 2 + 4 * img_n)
        img2[img_n] = ax2[img_n].imshow(np.zeros([img_side, img_side]),
                                        vmin=0, vmax=1)

        # test plots

        # init plots for target mnist digits
        ax3[img_n] = fig1.add_subplot(tests, 4, 3 + 4 * img_n)
        img3[img_n] = ax3[img_n].imshow(np.zeros([img_side, img_side]),
                                        vmin=0, vmax=1)

        # init plots for reproduced mnist digit
        ax4[img_n] = fig1.add_subplot(tests, 4, 4 + 4 * img_n)
        img4[img_n] = ax4[img_n].imshow(np.zeros([img_side, img_side]),
                                        vmin=0, vmax=1)

    fig2 = plt.figure(figsize=(10, 5))

    # init plot for errors
    ax3 = fig2.add_subplot(111)
    error_plot, = ax3.plot(0, 0)
    error_points = ax3.scatter(0, 0)

    ax3.set_xlim([-10, n_epochs + 10])
    ax3.set_ylim([0, 0.08])

    # main loop

    # vector for epoch errors storage
    errsums = np.zeros(n_train)
    # indices of the training and test patterns (for shuffling)
    train_idcs = range(n_train)
    test_idcs = range(n_test)

    # epochs
    for k in xrange(n_epochs):
        # for each epoch a different random
        #      order of pattern presentation
        random.shuffle(train_idcs)
        # storage of errors for each pattern
        errors = np.zeros(n_train)

        # iterate over training patterns
        for t in train_idcs:
            # we rescale values in each pattern so that pixels are
            #      between 0 an 1
            input_pattern = np.array(mndata.train_images[t]) / 255.0
            # the network spreading and learning
            bp.step(input_pattern)
            bp.learn(input_pattern)
            # save error for this pattern
            errors[t] = bp.get_mse()

        # save the error mean of the finished epoch
        errsums[k] = np.mean(errors)

        # train samples
        train_inps = []
        train_outs = []
        # for each epoch a different random
        #      order of tests presentation
        random.shuffle(test_idcs)
        for img_n in xrange(tests):
            inp = np.array(mndata.train_images[test_idcs[img_n]]) / 255.0
            bp.step(inp)
            out = bp.units[-1].copy()
            train_inps.append(inp)
            train_outs.append(out)

        # test samples
        test_inps = []
        test_outs = []
        # for each epoch a different random
        #      order of tests presentation
        random.shuffle(test_idcs)
        for img_n in xrange(tests):
            inp = np.array(mndata.test_images[test_idcs[img_n]]) / 255.0
            bp.step(inp)
            out = bp.units[-1].copy()
            test_inps.append(inp)
            test_outs.append(out)

        # plots
        if k == 0:
            ax3.set_ylim([0, errsums.max() * 1.2])
        for img_n in xrange(tests):
            img1[img_n].set_data(test_inps[img_n].reshape(img_side, img_side))
            img2[img_n].set_data(test_outs[img_n].reshape(img_side, img_side))
            img3[img_n].set_data(train_inps[img_n].reshape(img_side, img_side))
            img4[img_n].set_data(train_outs[img_n].reshape(img_side, img_side))

        error_plot.set_data(np.arange(1, k + 2), errsums[:(k + 1)])
        error_points.set_offsets(np.vstack([np.arange(1, k + 2),
                                            errsums[:(k + 1)]]).T)

        # save log
        np.savetxt("log",
                   ["epoch: {:d}  error: {:7.5f}\n".format(k, errsums[k])],
                   fmt="%s")
        # save figs
        fig1.savefig("digits.jpg")
        fig2.savefig("errors.jpg")

def bp_mnist_test():

    n_mnist_pixels = 28*28

    # init backprop object
    bp = BackProp(
        outfun=sigmfun, derfun=sigmder,
        n_units_per_layer=[n_mnist_pixels,
                           100, 10, 100,
                           n_mnist_pixels],
        eta=0.01)

    mnist_test(bp)



if __name__ == "__main__":

    # import matplotlib.pyplot as plt
    #
    # plt.plot(xor_test())
    # plt.show()

    mnist_test()
