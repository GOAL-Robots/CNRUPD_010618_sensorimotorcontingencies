import os
import random
import numpy as np
import matplotlib.pyplot as plt
from backprop import BackProp, sigmfun, sigmder 

np.set_printoptions(suppress=True,
                    precision=5,
                    linewidth=10000)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def mnist_test(bp, plotter):
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
        plotter.step(errsums, test_inps, test_outs, train_inps, train_outs)


        
class Plotter(object):
     
    def __init__(self, print_figs = False):
        
        self.print_figs = print_figs
        # number of tests
        self.tests = 8
        # The number of pixels per side of all images
        self.img_side = 28  
        # number of epochs
        self.n_epochs = 300
     
        self.fig1 = plt.figure(figsize=(8, 12))
        # init plots for target mnist digits
        (self.ax1, self.img1,
         self.ax2, self.img2, 
         self.ax3, self.img3, 
         self.ax4, self.img4) = (dict() for x in xrange(8))
        
        for img_n in xrange(self.tests):

            # train plots
    
            # init plots for target mnist digits
            self.ax1[img_n] = self.fig1.add_subplot(self.tests, 4, 1 + 4 * img_n)
            self.img1[img_n] = self.ax1[img_n].imshow(np.zeros([self.img_side, self.img_side]),
                                            vmin=0, vmax=1)
    
            # init plots for reproduced mnist digit
            self.ax2[img_n] = self.fig1.add_subplot(self.tests, 4, 2 + 4 * img_n)
            self.img2[img_n] = self.ax2[img_n].imshow(np.zeros([self.img_side, self.img_side]),
                                            vmin=0, vmax=1)
    
            # test plots
            # init plots for target mnist digits
            self.ax3[img_n] = self.fig1.add_subplot(self.tests, 4, 3 + 4 * img_n)
            self.img3[img_n] = self.ax3[img_n].imshow(np.zeros([self.img_side, self.img_side]),
                                            vmin=0, vmax=1)
    
            # init plots for reproduced mnist digit
            self.ax4[img_n] = self.fig1.add_subplot(self.tests, 4, 4 + 4 * img_n)
            self.img4[img_n] = self.ax4[img_n].imshow(np.zeros([self.img_side, self.img_side]),
                                            vmin=0, vmax=1)

        self.fig2 = plt.figure(figsize=(5, 2.5))

        # init plot for errors
        self.ax3 = self.fig2.add_subplot(111)
        self.error_plot, = self.ax3.plot(0, 0)
        self.error_points = self.ax3.scatter(0, 0)

        self.ax3.set_xlim([-10, self.n_epochs + 10])
        self.ax3.set_ylim([0, 0.08])
        
        self.k = 0
    
    def step(self, errsums, test_inps, test_outs, train_inps, train_outs):
        
        k = self.k
        
        if k  == 0:
            self.ax3.set_ylim([0, errsums.max() * 1.2])
        for img_n in xrange(self.tests):
            self.img1[img_n].set_data(test_inps[img_n].reshape(self.img_side, self.img_side))
            self.img2[img_n].set_data(test_outs[img_n].reshape(self.img_side, self.img_side))
            self.img3[img_n].set_data(train_inps[img_n].reshape(self.img_side, self.img_side))
            self.img4[img_n].set_data(train_outs[img_n].reshape(self.img_side, self.img_side))

        self.error_plot.set_data(np.arange(1, k + 2), errsums[:(k + 1)])
        self.error_points.set_offsets(np.vstack([np.arange(1, k + 2),
                                            errsums[:(k + 1)]]).T)
        if self.print_figs == True:
            # save log
            np.savetxt("log",
                       ["epoch: {:d}  error: {:7.5f}\n".format(k, errsums[k])],
                       fmt="%s")
            # save figs
            self.fig1.savefig("digits.jpg")
            self.fig2.savefig("errors.jpg")
        else:
            plt.pause(0.1)
        
        print k
        self.k += 1
        
            

if __name__ == "__main__":

    plt.ion()
    
    n_mnist_pixels = 28*28

    # init backprop object
    bp = BackProp(
        outfun=sigmfun, derfun=sigmder,
        n_units_per_layer=[n_mnist_pixels,
                           100, 10, 100,
                           n_mnist_pixels],
        eta=0.01)

    plotter = Plotter()
    mnist_test(bp, plotter)


