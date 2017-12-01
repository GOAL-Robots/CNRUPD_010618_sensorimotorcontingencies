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
    """ Keep track of the involvement of each hidden unit 
        in the optimization of error  
    """

    def __init__(self, hidden_units, eta=0.1, alpha=0.05):
        """
        :param hidden_units: list(np.array). the hidden layers of a network
        :param eta: float. The learning rate of the prediction optimization
        :param alpha: float. The slope of the error-thresholding function 
                      (see step)
        """
        self.hidden_units = hidden_units
        self.n = len(np.hstack(self.hidden_units))
        self.w = np.zeros(self.n)
        self.eta = eta
        self.alpha = alpha
        

    def step(self, errors):
        """ A single step of update of the prediction of the error of the 
            network
        :param errors: np.array. The vector of errors of the output layer
                of the network form the related target 
        """
        units = np.hstack(self.hidden_units)
        self.prediction = np.dot(units, self.w)
        self.match = 1.0 - np.tanh(mse(errors) * self.alpha)
        self.w += self.eta * (self.prediction - self.match) * units
        return self.predict, self.match

class BackPropMod(BackProp):
    """ Backpropagation with prediction 
    """
    def __init__(self, *args, **kargs):
        super(BackPropMod, self).__init__(*args, **kargs)
        self.predictor = Predictor(self.units[1:])
    
    def modulate_deltas(self):
        predictions = [ self.predictor.w[ layer:(layer + units_per_layer)] 
                       for layer, units_per_layer 
                       in zip(
                           range(self.n_layers - 1),
                           self.n_units_per_layer[1:])]
        
        for i in range(self.n_layers - 1):
            self.deltas[i+1] *= 1 - predictions[i]
            
    def learn(self, target):
        """ Overloading of the learning step
        :param target: np.array. The desired output for the current input
        """
        self.error_backprop(target)
        self.update_weights()
        self.predictor.step(self.error)
        
#------------------------------------------------------------------------------ 

if __name__ == "__main__":

    n_mnist_pixels = 28*28
    
    # init backprop object
    bp = BackPropMod(
        outfun=sigmfun, derfun=sigmder,
        n_units_per_layer=[n_mnist_pixels,
                           100, 10, 100,
                           n_mnist_pixels],
        eta=0.01)
    print "Done"
