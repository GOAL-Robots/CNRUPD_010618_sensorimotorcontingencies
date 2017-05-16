import os
import random
import numpy as np

# ----------------------------------------------------------------------------
# utils ----------------------------------------------------------------------

# Add a bias unit to the input
def biased(x) :
    return np.hstack([1,x])

# sigmoid function
# t   float temperature
def sigmfun(x, t = 1.0) :
    return 1.0/(1.0 + np.exp(-x/t))

# sigmoid derivative
def sigmder(y) :
    return y*(1-y)

def relu(x) :
    return np.maximum(0.0, x)

def reluder(x) :
    return (x>0.0)*1.0

# create a numpy random number generator
def make_random_generator(SEED = None):
    if SEED is None:
        SEED = np.fromstring(os.urandom(4), dtype=np.uint32)[0]
    return np.random.RandomState(SEED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

class BackProp:
    """
    Error back propagation algorithm for
    the learning of the weights of a multi-layered perceptron.
    """

    def __init__(self, outfun = sigmfun,
                 derfun =  sigmder, eta = 0.01,
                 n_units_per_layer = [2, 2, 1],
                 rng = make_random_generator()):
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
        for n in self.n_units_per_layer :
            self.units.append(np.zeros(n))

        # Initialize self.deltas
        self.deltas = []
        for n in self.n_units_per_layer :
            self.deltas.append(np.zeros(n))

        # Initialize weights
        self.weights = []
        for n_out, n_inp in zip(self.n_units_per_layer[1:],
                                self.n_units_per_layer[:-1]):
            self.weights.append(np.zeros([n_out, n_inp + 1]))


    def step(self, input_pattern):
        """
        :param  input_pattern  the current input pattern
        """
        # spreading

        # linear function
        self.units[0] = input_pattern

        # iterate deep layers
        for layer in xrange(self.n_layers - 1) :

            # Bias-plus-input vector
            input_pattern =  np.dot(self.weights[layer],
                                 biased(self.units[layer]))

            # nonlinear function
            self.units[layer + 1] = self.outfun(input_pattern)


    def learn(self, target):
        """
        :param  target  the desired output for the current input
        """

        #---------------------------------------------------------------------
        # error back-propagation

        # the initial outer-error is the real one
        # (expected vs real outputs)
        
        self.error = target - self.units[-1]

        self.deltas[-1] = self.error * \
                self.derfun(self.units[-1])
        # iterate layers backward
        for layer in  xrange(self.n_layers-1, 0, -1) :
            # for each layer evaluate the
            # back-propagation error
            w_T = (self.weights[layer-1][:, 1:]).T
            self.deltas[layer - 1] = \
                    np.dot(w_T, self.deltas[layer]) * \
                    self.derfun(self.units[layer - 1])

        #---------------------------------------------------------------------
        # weight updates

        for layer in  xrange(self.n_layers-1):
            # error plus derivative
            self.weights[layer] += self.eta * np.outer(               
                    self.deltas[layer + 1],
                    biased(self.units[layer]))

    def get_mse(self):
        return 0.5*(np.mean(self.error**2))


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# test
def xor_test():

    bp = BackProp(
        outfun = sigmfun, derfun = sigmder,
        n_units_per_layer = [2,  2, 1],
        eta = 1)

    data =np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]])
    idcs = range(4)
    trials = 20000

    for t in xrange(trials):
        random.shuffle(idcs)
        errors = np.zeros(4)
        for p in idcs:
            bp.step(data[p,:2])
            bp.learn(data[p,2])
            errors[p] = bp.get_mse()
            print data[p], bp.units[-1], errors[p]
        print
        print np.mean(errors)
        print

def mnist_test():

    np.set_printoptions(suppress = True,
                        precision = 2,
                        linewidth = 10000)

    # import matplotlib.pyplot as plt
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # line, = ax.plot(0)
    # ax.set_xlim([-10, 3e2+10])
    # ax.set_ylim([0, 1.2e-1])

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
    n_mnist_pixels = img_side*img_side

    n_train = len(mndata.train_images)
    n_test = len(mndata.test_images)

    bp = BackProp(
        outfun = sigmfun, derfun = sigmder,
        n_units_per_layer = [n_mnist_pixels,  10, 10],
        eta= 0.01)

    n_trials = 300
    errsums = np.zeros(n_train)
    idcs = range(n_train)
    for k in xrange(n_trials):
        random.shuffle(idcs)
        errors = np.zeros(n_train)
        rng = idcs
        for t in rng:

            input_pattern = np.array(mndata.train_images[t])
            target = np.zeros(10)
            target[mndata.train_labels[t]] = 1.0

            bp.step(input_pattern)
            bp.learn(target)
            errors[t] = bp.get_mse()

        errsums[k] = np.mean(errors[rng]) 
        print k, errsums[k]
        # line.set_data(10+np.arange(k), errsums[:k])
        # plt.pause(0.01)

if __name__ == "__main__":

    xor_test()

