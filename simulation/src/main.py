#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import sys
import copy
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

## Start Qt event loop unless running in interactive mode.
def main(args):
     
    GRAPHICS = bool(args.graphics) 
    STIME = int(args.stime)  
    SDIR = args.save_dir
    if SDIR[-1]!='/': SDIR += '/'
     
    DUMP = int(args.dump) 
    LOAD = int(args.load) 
   
    log_sensors = open(SDIR+"log_sensors", "w")
    # log_cont_sensors = open(SDIR+"log_cont_sensors", "w")
    log_position = open(SDIR+"log_position", "w")
    log_predictions = open(SDIR+"log_predictions", "w")
    log_targets = open(SDIR+"log_targets", "w")
    log_weights = open(SDIR+"log_weights", "w")
    

    dumpfile = SDIR+"dumped_robot"
    
    if LOAD :
        print "loading ..."
        with gzip.open(dumpfile, 'rb') as f:
            robot = pickle.load(f)
    else :
        robot = model.Robot()

    robot.log_sensors = log_sensors
    # robot.log_cont_sensors = log_cont_sensors
    robot.log_position = log_position
    robot.log_predictions = log_predictions
    robot.log_targets = log_targets
    robot.log_weights = log_weights
    
    print "simulating ..."
    if GRAPHICS :

        from model import plotter 
        plotter.graph_main(robot)
        
    else:
        bar = progressbar.ProgressBar( 
                maxval=STIME, 
                widgets=[progressbar.Bar('=', '[', ']'), 
                    ' ', progressbar.Percentage()],
                term_width=30)
        
        bar.start()
        for t in range(STIME):
            robot.step()
            bar.update(t+1)
        bar.finish()

    if DUMP :
        
        print "dumping ..."
        with gzip.open(dumpfile, 'wb') as f:
            robot.init_streams()
            robot = pickle.dump(robot, f)


if __name__ == "__main__" :

    import argparse
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-g','--graphics',
            help="Graphics on",
            action="store_true", default=False) 
    parser.add_argument('-d','--dump',
            help="dump the robot object",
            action="store_true", default=False) 
    parser.add_argument('-l','--load',
            help="load the robot object",
            action="store_true", default=False) 
    parser.add_argument('-s','--save_dir',
            help="storage directory",
            action="store", default=os.getcwd())      
    parser.add_argument('-t','--stime',
            help="Simulation time (only for graphics off)",
            action="store", default=2000)  
    args = parser.parse_args()

    import cProfile, pstats, StringIO
    
    main(args)
    
    #### profiling
    # pr = cProfile.Profile()
    # pr.enable()
    # 
    # main(args)
    # 
    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()

