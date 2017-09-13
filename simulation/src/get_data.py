#!/usr/bin/python
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
            simulation = pickle.load(f)
        #TODO:extract and save data   
    except :  
        raise Exception("{} does not exist".format(dumpfile))

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

