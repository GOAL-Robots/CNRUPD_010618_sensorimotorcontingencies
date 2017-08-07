#!/usr/bin/env python

import sys

import numpy as np
import time
import os

from gauss_utils import *

#------------------------------------------------------------
GaussianMaker = OptimizedGaussianMaker

class Kohonen(object) :

    def __init__(self,
            stime               = 400,
            n_dim_out           = 1,
            bins                = 1,
            n_output            = 25,
            n_input             = 10,
            neighborhood        = 30,
            neighborhood_decay  = 300,
            neighborhood_bl     = 1.0,
            eta                 = 0.1,
            eta_decay           = 300,
            eta_bl              = 0.001,
            weight_bl           = 0.00,
            average_decay       = 0.1,
            normalize           = lambda x : x,
            rng                 = np.random.RandomState(int(time.time()))
            ) :
        """
            :param stime: time of simulation
            :param n_dim_out: number of dimensions of output topology
            :param bins: list of bins for each dimension
            :param n_output: number of output units
            :param n_input: number of input elements
            :param neighborhood: radius of neighbor-to-winner output units
            :param neighborhood_decay: neighborhood decay
            :param neighborhood_bl: neighborhood bl
            :param eta: learning rate
            :param eta_decay: learning rate decay
            :param average_decay: decat of the raw output moving average
            :param normalize: normalizing function
            :param rng: random number generator
            :type stime:
            :type n_dim_out:
            :type bins : list(int)
            :type n_output:
            :type n_input:
            :type neighborhood:
            :type neighborhood_decay:
            :type neighborhood_bl:
            :type eta:
            :type eta_decay:
            :type average_decay:
            :type normalize:
            :type rng: (numpy.random.RandomState)
      """

        # time-step counter
        self.t = 0

        self.STIME = stime
        self.ETA = eta
        self.ETA_BL = eta_bl
        self.ETA_DECAY = eta_decay
        self.N_DIM_OUT = n_dim_out
        self.normalize = normalize
        self.AVERAGE_DECAY = average_decay

        if np.isscalar(bins) :
            self.BINS = np.ones(self.N_DIM_OUT)*bins
        else :
            self.BINS = bins

        self.N_OUTPUT = n_output
        self.N_INPUT = n_input
        self.neighborhood = neighborhood
        self.neighborhood_DECAY = neighborhood_decay
        self.neighborhood_BL = neighborhood_bl

        self.inp = np.zeros(self.N_INPUT)
        self.out = np.zeros(self.N_OUTPUT)
        self.out_raw = np.zeros(self.N_OUTPUT)
        self.inp_min = 0
        self.inp_max = 0


        lims = [ [0,self.BINS[x]-1,self.BINS[x]]
                for x in xrange(self.N_DIM_OUT) ]
        self.gmaker = GaussianMaker(lims)

        # timing
        self.t = 0

        # initial weights
        self.inp2out_w = weight_bl*rng.randn(self.N_OUTPUT,self.N_INPUT)

        # data storage
        self.data = dict()
        self.l_inp = 0
        self.l_out = 1
        self.l_out_raw = 2
        self.data[self.l_inp] = np.zeros([self.N_INPUT,self.STIME])
        self.data[self.l_out] = np.zeros([self.N_OUTPUT,self.STIME])
        self.data[self.l_out_raw] = np.zeros([self.N_OUTPUT,self.STIME])

    def step(self, inp, neigh_scale = None) :
        """ Spreading

        :param inp: cueerent input vector
        :type inp: array(n_input, float)

        """

        self.t += 1

        # # input
        if not all(inp==0) :

            x = self.normalize(inp)
            self.inp = x
            w = self.inp2out_w
            y = np.dot(w,x) -0.5*np.diag(np.dot(w,w.T))

            # Calculate neighbourhood
            #   Current neighbourhood
            curr_neighborhood = self.updateNeigh(neigh_scale)

            max_index = np.argmax(y) # index of maximum
            self.idx = max_index
            
            # output:
            self.out_raw = y
            point = map1DND(max_index, self.N_DIM_OUT, self.BINS)
            self.out,_ = self.gmaker(point,
                    np.ones(self.N_DIM_OUT)*(
                        (np.maximum(curr_neighborhood, 0.0e-6))**2))
        else:
            x = np.zeros(inp.shape)
            self.out = np.zeros(self.N_OUTPUT)

    def updateEta(self, value = None):
        '''
        Update the learning rate. It allows to manually update current eta value

        :param  value   the current decay (if None it is the negative exponential of
                        ETA_DECAY in time)
        :type   value   float

        '''

        eta = None

        if value is None:
            eta = self.ETA_BL + self.ETA* np.exp(-self.t/self.ETA_DECAY)
        else:
            eta = self.ETA_BL +value*(self.ETA)

        return eta

    def updateNeigh(self, value=None):
        '''
        Update the neighborhood radius. It allows to manually update current
        neighborhood radius value

        :param  value   the current neighborhood radius (if None it is the
                        negative exponential of neighborhood_DECAY in time)
        :type   value   float or array

        '''

        neighborhood = None

        if value is None:
            neighborhood = self.neighborhood_BL + \
                self.neighborhood * np.exp(-self.t / self.neighborhood_DECAY)
        else:
            neighborhood = self.neighborhood_BL + value * (self.neighborhood)

        return neighborhood.copy()



    def learn(self, eta_scale = None, pred = None) :
        """
        Learning step

        :param  eta_scale   the value of current eta decay (see updateEta)
        :param  pred        the current amount of learning of each output receptive field
        :type   eta_scale   float
        :type   pred        array(float)

        """

        if pred is None:
            pred = np.ones(self.N_OUTPUT)

        assert( len(pred) == self.N_OUTPUT )

        eta = self.updateEta(eta_scale)

        # Update weights
        x = self.inp
        y = self.out if self.neighborhood > 0 \
                else self.out*(self.out == self.out.max())
        w = self.inp2out_w
        
        w += eta* (np.outer(y * pred,x) -  
                np.outer(y * pred, np.ones(self.N_INPUT)) * w)

    def store(self):
        """ storage """

        tt = self.t%self.STIME
        self.data[self.l_inp][:,tt] = self.inp
        self.data[self.l_out][:,tt] = self.out

        out = self.out_raw*2
        datax = self.data[self.l_out_raw]
        win = self.idx

    def reset_data(self):
        """ Reset """

        for k in self.data :
            self.data[k] = self.data[k]*0
