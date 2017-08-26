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
from model.plotter import reshape_weights, angles2positions

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
    weights = reshape_weights(weights)
    np.savetxt("weights", weights)


    plt.figure()
    rng = np.linspace(-0.5, 99.5, 6)
    plt.imshow(weights[::-1].T[::-1], aspect="auto", cmap=plt.cm.binary)
    for y in rng:
        for x in rng:
            plt.plot([-0.5,99.5], [y,y], color="black")
            plt.plot([x,x], [-0.5,99.5], color="black")
    plt.savefig(SDIR+"/weights.png")        
            
    plt.figure() 
    act = KinematicActuator()
    for idx,pos in simulation.gs.target_position.iteritems():
        plt.subplot(5,5,idx+1, aspect="equal")
        positions = angles2positions([pos], act)[0]
        plt.plot(*positions.T)
        plt.scatter(*positions.T, s=10)
        plt.xlim([-4,4])
        plt.ylim([-.1,3])
        plt.axis("off") 
    plt.savefig(SDIR+"/positions.png")        

if __name__ == "__main__" :

    import argparse
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-s','--save_dir',
            help="storage directory",
            action="store", default=os.getcwd())      
    args = parser.parse_args()

    main(args)

