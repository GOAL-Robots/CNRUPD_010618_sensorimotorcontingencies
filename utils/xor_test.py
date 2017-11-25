import os
import random
import numpy as np
import matplotlib.pyplot as plt
from backprop import BackProp, sigmfun, sigmder 

np.set_printoptions(suppress=True,
                    precision=5,
                    linewidth=10000)


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


if __name__ == "__main__":

    """  Classical xor example (using BackProp)  """
    # init the backprop object
    bp = BackProp(
        outfun=sigmfun, derfun=sigmder,
        n_units_per_layer=[2, 2, 1],
        eta=1)

    errors = xor_test(bp)
    plt.plot(errors)
    plt.show()