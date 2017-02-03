#!/bin/bash

set -e

# first of all detect if we are executing ithin the right folder
CURR_DIR=$(pwd)
[ ! -f src/model/Robot.py ] && (echo "you must execute within the project folder"; exit)



# Manage arguments
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

usage()
{
cat << EOF

usage: $0 options

This script runs the robot simulation in batch mode and collects data 

OPTIONS:
   -t --stime       number of timesteps of a single simulation block
   -w --wdir        working directory
   -s --start       dumped_robot to start from
   -n --n_blocks    number of simulation blocks
   -h --help        show this help

EOF
}



WORK_DIR=
START=
STIME=100000
N_BLOCKS=1

# getopt
GOTEMP="$(getopt -o "t:w:n:s:h" -l "stime:,wdir:,n_blocks:,start:,help"  -n '' -- "$@")"

if ! [ "$(echo -n $GOTEMP |sed -e"s/\-\-.*$//")" ]; then
    usage; exit;
fi

eval set -- "$GOTEMP"

while true ;
do
    case "$1" in
        -t | --stime) 
            STIME="$2"
            shift 2;;
        -w | --wdir) 
            WORK_DIR="$2"
            shift 2;;
        -n | --n_blocks) 
            N_BLOCKS="$2"
            shift 2;;
        -s | --start) 
            START="$2"
            shift 2;;
        -h | --help)
            echo "on help"
            usage; exit;
            shift;
            break;;
        --) shift ; 
            break ;;
    esac
done

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

#################################################################################

# the execution command
CMD="python $CURR_DIR/src/main.py"

# clean working dir
[ ! -e $WORK_DIR ] && mkdir $WORK_DIR
rm -fr $WORK_DIR/*

# run  n-th blocks
for((n=0;n<$[N_BLOCKS];n++)); 
do
    
    snum="$(printf "%06d" $n)"

    # run first block
    if [ $n -eq 0 ]; then
        
        if [ ! -z $START ]; then  # THERE IS a previous dump from which to start 

            cp $START $WORK_DIR 
            $CMD -t $STIME -d -l -s $(pwd)/$WORK_DIR

        else    # NO previous dumps from which to start
            
            $CMD -t $STIME -d -s $(pwd)/$WORK_DIR

        fi

    else

        # run the following n-th block
        $CMD -t $STIME -d -l -s $(pwd)/$WORK_DIR

    fi
    
    # store block
    tag=$(date +%Y%m%d%H%M%S) 
    for f in $WORK_DIR/*; 
    do
        cp $f store/"$(basename $f)-$tag"
    done

done
echo done
