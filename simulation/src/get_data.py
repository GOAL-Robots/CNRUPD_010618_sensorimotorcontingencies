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

import model

## Start Qt event loop unless running in interactive mode.
def main(args):
     
    SDIR = args.save_dir
    if SDIR[-1]!='/': SDIR += '/'
     
    dumpfile = SDIR+"dumped_robot"
    
    if os.path.exists(dumpfile) :
    
        print "loading ..."
        with gzip.open(dumpfile, 'rb') as f:
            simulation = pickle.load(f)
    
        #TODO:extract and save data
    
    else :
        
        raise Exception("{} does not exist".format(dumpfile))



if __name__ == "__main__" :

    import argparse
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-s','--save_dir',
            help="storage directory",
            action="store", default=os.getcwd())      
    args = parser.parse_args()

    main(args)

