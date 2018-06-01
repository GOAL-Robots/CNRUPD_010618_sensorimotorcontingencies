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
import numpy as np
import matplotlib.pyplot as plt

class Oscillator(object):
    def __init__(self, scale, freqs, res = None):
        '''
        :param scale: scaling factor for the frequency
        :type scale: float

        :param freqs: list of frequencies (one for each trjectory)
        :type freqs: list(float)
        '''

        self.scale = scale
        self.freqs = np.array(freqs)
        self.trajectories_num = self.freqs.size  
        
        self.t = np.random.randint(0,40)

        self.res = 0 if res is None else res

    def __call__(self):
        """ Calls an oscillator's step

        :return: the current angles after one step
        :rtype: numpy.array(trajectories_num, dtype=float)
        """
        
        self.res += 0.001*(-self.res -(0.006**-1)*np.sin(2.0 * np.pi * self.freqs * self.t / 
                       self.scale + np.pi)) * 0.5
        self.res = np.tanh(self.res)
        self.t += 1

        return self.res

    def reset(self):

        self.t = 0

plt.close("all")

stime = 100
f_n = 6

o = Oscillator(6, np.random.uniform(-1,1,f_n), np.random.uniform(-1,1,f_n))

y = np.zeros([f_n,stime])
for t in range(stime):
    if t == stime/2:
        res = o.res.copy()
        o = Oscillator(6, 
                np.random.uniform(-1,1,f_n), 
                res=res)
    y[:, t] = o()

plt.plot(y.T)
plt.show()
