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


import os
import sys
import cPickle as pickle
import gzip

# working dir is the base dir of this file
pathname = os.path.dirname(sys.argv[0])
if pathname: os.chdir(pathname)

################################################################
################################################################
# To force stop on exceptions

import traceback
def my_excepthook(type, value, tback): 
    traceback.print_exception(type,value,tback) 
    sys.exit(1)

sys.excepthook = my_excepthook

#################################################################
#################################################################

import progressbar
import model
import numpy as np

## Start Qt event loop unless running in interactive mode.
def main(args):
     
    GRAPHICS = bool(args.graphics) 
    STIME = int(args.stime)  
    SDIR = args.save_dir
    if SDIR[-1]!='/': SDIR += '/'
     
    DUMP = int(args.dump) 
    LOAD = int(args.load) 
    SEED = int(args.seed) if args.seed is not None else None
   
    log_sensors = open(SDIR+"log_sensors", "w")
    log_cont_sensors = open(SDIR+"log_cont_sensors", "w")
    log_position = open(SDIR+"log_position", "w")
    log_predictions = open(SDIR+"log_predictions", "w")
    log_targets = open(SDIR+"log_targets", "w")
    log_weights = open(SDIR+"log_weights", "w")
    log_trials = open(SDIR+"log_trials", "w")
    

    dumpfile = SDIR+"dumped_robot"
    
    if LOAD :
        print "loading ..."
        with gzip.open(dumpfile, 'rb') as f:
            (simulation, state) = pickle.load(f)
            rng = np.random.RandomState(simulation.seed)  
            rng.set_state(state)
            simulation.rng = rng
                
    else :
        if SEED is not None:
            rng = np.random.RandomState(SEED)  
            simulation = model.Simulation(rng)
        else :
            simulation = model.Simulation()
            SEED = simulation.seed

        with open(SDIR+"seed", "w") as f:
            f.write("%d" % SEED)

    simulation.log_sensors = log_sensors
    simulation.log_cont_sensors = log_cont_sensors
    simulation.log_position = log_position
    simulation.log_predictions = log_predictions
    simulation.log_targets = log_targets
    simulation.log_weights = log_weights
    simulation.log_trials = log_trials
    
    print "simulating ..."
    if GRAPHICS :

        from model import plotter 
        plotter.graph_main(simulation)
        
    else:
        bar = progressbar.ProgressBar( 
                maxval=STIME, 
                widgets=[progressbar.Bar('=', '[', ']'), 
                    ' ', progressbar.Percentage()],
                term_width=30)
        
        bar.start()
        for t in range(STIME):
            simulation.step()
            bar.update(t+1)
        bar.finish()

    if DUMP :
        
        print "dumping ..."
        with gzip.open(dumpfile, 'wb') as f:
            state = simulation.rng.get_state()
            simulation.init_streams()
            pickle.dump((simulation, state), f)
            


if __name__ == "__main__" :

    import argparse
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-g','--graphics',
            help="Graphics on",
            action="store_true", default=False) 
    parser.add_argument('-f','--prof',
            help="profiling",
            action="store_true", default=False) 
    parser.add_argument('-d','--dump',
            help="dump the simulation object",
            action="store_true", default=False) 
    parser.add_argument('-l','--load',
            help="load the simulation object",
            action="store_true", default=False) 
    parser.add_argument('-s','--save_dir',
            help="storage directory",
            action="store", default=os.getcwd())      
    parser.add_argument('-t','--stime',
            help="Simulation time (only for graphics off)",
            action="store", default=2000)  
    parser.add_argument('-S','--seed',
            help="Seed of the simulation",
            action="store")  
    args = parser.parse_args()

    import cProfile, pstats, StringIO
     

    if args.prof == True:

        #### profiling
        pr = cProfile.Profile()
        pr.enable()
        
        main(args)
        
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
    else: 
        main(args)

