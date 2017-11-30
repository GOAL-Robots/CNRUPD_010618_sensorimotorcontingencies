import os
import random
import numpy as np
from backprop import BackProp, sigmfun, sigmder
np.set_printoptions(suppress=True,
                    precision=5,
                    linewidth=10000)

#------------------------------------------------------------------------------ 

class Predictor(object):

    def __init__(self, hidden_units, eta=0.1):
        self.n = len(np.hstack(hidden_units))
        self.w = np.zeros(self.n)
        self.eta = eta

    def step(hidden_units, errors):
        units = np.hstack(hidden_units)
        
        w += self.eta*

class BackPropMod(BackProp):
    
    def __init__(self, *args, **kargs):
        super(BackPropMod, self).__init__(*args, **kargs)
        self.predictor = Predictor(1)
        
    def learn(self, target):
        """
        :param  target  the desired output for the current input
        """

        self.error_backprop(target)
        self.update_weights()
        
#------------------------------------------------------------------------------ 

if __name__ == "__main__":

    n_mnist_pixels = 28*28
    
n_mnist_pixels = 28*28

# init backprop object
bp = BackPropMod(
    outfun=sigmfun, derfun=sigmder,
    n_units_per_layer=[n_mnist_pixels,
                       100, 10, 100,
                       n_mnist_pixels],
    eta=0.01)
print "Done"
