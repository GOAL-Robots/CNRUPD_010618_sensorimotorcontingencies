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


def hist_comp(x):
    h = np.histogram(x) 
    freqs = h[0]
    centr = np.linspace(h[1][0],h[1][-1], 
            2*len(h[1])-1 )[1::2]
    return np.vstack([freqs, centr]).T


#################################################################
#################################################################

def reshape_weights(w):
    reshaped_w = []
    reshaped_w_raw = []
    single_w_raws = int(np.sqrt(len(w[0])))
    single_w_cols = single_w_raws

    n_single_w = len(w)
    out_raws = np.sqrt(n_single_w)
    for single_w, i in zip(w, xrange(n_single_w)):
        reshaped_w_raw.append(
            single_w.reshape(single_w_cols,
                single_w_raws, order="F"))
        if (i+1)%out_raws == 0:
            reshaped_w.append(np.hstack(reshaped_w_raw))
            reshaped_w_raw =[]
    reshaped_w = np.vstack(reshaped_w)

    return reshaped_w

def reshape_coupled_weights(w):
    '''
    Reshape the matrix of weight from two (nxn) input layers
    to one (mxm) output layers so that it becomes a block matrix
    made by  mxm  nx2n matrices
    '''

    reshaped_w = []
    reshaped_w_raw = []
    single_w_raws = int(np.sqrt(len(w[0])/2))
    single_w_cols = 2*single_w_raws

    n_single_w = len(w)
    out_raws = np.sqrt(n_single_w)
    for single_w, i in zip(w, xrange(n_single_w)):
        reshaped_w_raw.append(
            single_w.reshape(single_w_cols,
                single_w_raws, order="F"))
        if (i+1)%out_raws == 0:
            reshaped_w.append(np.hstack(reshaped_w_raw))
            reshaped_w_raw =[]
    reshaped_w = np.vstack(reshaped_w)

    return reshaped_w

def angles2positions(angles_array, act = None):
    
    if act is None:
        act = KinematicActuator()
    
    positions = []
    num_angles = len(angles_array[0])
    for angles in angles_array:
        l_angles = angles[:(num_angles/2)]
        r_angles = angles[(num_angles/2):]
        l_angles, r_angles = act.rescale_angles(l_angles, r_angles)
        act.set_positions_basedon_angles(l_angles, r_angles)
    
        positions.append(np.vstack((act.position_l[::-1], act.position_r)))
    
    return positions

def expSaturatedDecay(x, temp = 50.0):
    # ret = 2.0 / (np.exp(-temp * x) + 1.0) - 1.0
    ret = 1.0*(x>1.0/temp) 
    return ret
        
