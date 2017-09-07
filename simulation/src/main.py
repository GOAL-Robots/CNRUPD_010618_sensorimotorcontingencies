#!/usr/bin/python
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
    

    dumpfile = SDIR+"dumped_robot"
    
    if LOAD :
        print "loading ..."
        with gzip.open(dumpfile, 'rb') as f:
            simulation = pickle.load(f)
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
            simulation.init_streams()
            simulation = pickle.dump(simulation, f)


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

