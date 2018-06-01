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
"""

The MIT License (MIT)

Copyright (c) 2015 Francesco Mannella <francesco.mannella@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
from __future__ import division

import numpy as np
import numpy.random as rnd
np.set_printoptions(suppress=True, precision=5, linewidth=9999999)


class PID(object) :

    def __init__(self, n=1, dt=0.1, Kp=0.1, Ki=0.9, Kd=0.001 ):

        self.n = n
        self.dt = dt

        self.previous_error = np.zeros(n)
        self.integral = np.zeros(n)
        self.derivative = np.zeros(n)
        self.setpoint = np.zeros(n)
        self.output = np.zeros(n)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def reset(self):
        n = self.n
        self.previous_error = np.zeros(n)
        self.integral = np.zeros(n)
        self.derivative = np.zeros(n)
        self.setpoint = np.zeros(n)
        self.output = np.zeros(n)

    def step(self, measured_value, setpoint=None):

        if setpoint is not None:
            self.setpoint = np.array(setpoint)

        error = setpoint - measured_value
        self.integral =  self.integral + error*self.dt
        self.derivative = (error - self.previous_error)/self.dt
        self.output = self.Kp*error + \
                self.Ki*self.integral + \
                self.Kd*self.derivative

        self.previous_error = error

        return self.output
