#!/usr/bin/python
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
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('agg')

import os
import sys
import cPickle as pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

# working dir is the base dir of this file
pathname = os.path.dirname(sys.argv[0])
if pathname: os.chdir(pathname)

################################################################
################################################################

import model
from model.Simulator import KinematicActuator
from model.utils import reshape_weights, angles2positions

## Start Qt event loop unless running in interactive mode.
def main(args):
     
    SDIR = args.save_dir
    if SDIR[-1]!='/': SDIR += '/'
     
    dumpfile = SDIR+"dumped_robot"
    
    try :
        print "loading ..."
        with gzip.open(dumpfile, 'rb') as f:
            simulation,_ = pickle.load(f)
        #TODO:extract and save data   
    except :  
        raise Exception("{} does not exist".format(dumpfile))

    print "loaded"
    # --------------------------------------------------------
    
    weights = simulation.gm.goalrep_som.inp2out_w
    np.savetxt(SDIR+"/weights", weights)

    weights = reshape_weights(weights)
    weights = weights[::-1].T[::-1]
        
    act = KinematicActuator()
    pos_db = []
    for idx, pos in simulation.gs.target_position.iteritems():
        positions = angles2positions([pos], act)[0]
        pos_db.append(
            np.hstack((
            np.ones([len(positions), 1]) * idx,
            positions
            )) )
    pos_db = np.vstack(pos_db)
    np.savetxt(SDIR+"/positions", pos_db)

if __name__ == "__main__" :

    import argparse
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-s','--save_dir',
            help="storage directory",
            action="store", default=os.getcwd())      
    args = parser.parse_args()
    
    main(args)

