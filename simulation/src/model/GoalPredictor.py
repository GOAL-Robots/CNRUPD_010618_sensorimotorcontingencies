#!/usr/bin/env python
##
## Copyright (c) 2018 Francesco Mannella. 
## 
## This file is part of sensorimotor-contingencies
## (see https://github.com/GOAL-Robots/CNRUPD_010618_sensorimotorcontingencies).
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##

import sys

import numpy as np
import numpy.random as rnd

def match(abstaction_layer, selection_layer) :
    '''
    :param abstaction_layer
    :param selection_layer
    '''
    al = np.array(abstaction_layer).astype("float")
    gl = np.array(selection_layer).astype("float")

    return np.any( np.logical_and(al>0,gl>0) )

class GoalPredictor(object) :

    def __init__(self, n_goal_units, eta) :
        '''
        :param n_goal_units: number of units in the goal layer
        :param eta: learning rate
        '''
        
        self.N_GOAL_UNITS = n_goal_units
        self.ETA = eta

        self.inp_layer = np.zeros(self.N_GOAL_UNITS)
        self.w = np.zeros(self.N_GOAL_UNITS)
        self.goal_selection_vec = np.zeros(self.N_GOAL_UNITS)

        self.prediction_error = 0.0
        self.out = 0.0
        

    def step(self, goal_selection_vec):

        self.goal_selection_vec = goal_selection_vec
        self.out = np.dot(self.w,self.goal_selection_vec)
    
    def getCurrPred(self) :
        res = self.w[self.goal_selection_vec>0]
        if len(res) == 1:
            return np.asscalar(res)
        return 0

    def learn(self, match):
        
        self.w += self.ETA*self.goal_selection_vec*(match - self.out)
        self.prediction_error = np.maximum(0.0, match - self.out)

