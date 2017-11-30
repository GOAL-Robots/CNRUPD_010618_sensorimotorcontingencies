import os
import random
import numpy as np
from backprop import BackProp, sigmfun, sigmder
np.set_printoptions(suppress=True,
                    precision=5,
                    linewidth=10000)

#------------------------------------------------------------------------------ 

def mse(x, y):
    return np.mean(np.power(x - y, 2))

#------------------------------------------------------------------------------ 

class Predictor(object):

    def __init__(self, hidden_units, eta=0.1, alpha=0.05):
        self.n = len(np.hstack(hidden_units))
        self.w = np.zeros(self.n)
        self.eta = eta
        self.alpha = alpha
        

    def step(hidden_units, errors):
        units = np.hstack(hidden_units)
        self.prediction = np.dot(units, self.w)
        self.match = 1.0 - np.tanh(mse(errors)*self.alpha)
        self.w += self.eta*(self.prediction - match)*units
        return self.predict, self.match

class BackPropMod(BackProp):
    
    def __init__(self, *args, **kargs):
        super(BackPropMod, self).__init__(*args, **kargs)
        self.predictor = Predictor(self.units[:-1])
        
    def learn(self, target):
        """
        :param  target  the desired output for the current input
        """

        self.error_backprop(target)
        self.update_weights()
        self.predictor.step(self.units[:-1], self.error)
        
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
