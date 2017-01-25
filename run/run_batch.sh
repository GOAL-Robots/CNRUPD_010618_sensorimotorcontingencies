#!/bin/bash

set -e

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


CURR_DIR=$(pwd)
[ ! -f src/model/Robot.py ] && (echo "you must execute within the project directory"; exit)

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

if [ -z $OMP_NUM_THREADS ]; then

    echo "OMP_NUM_THREADS auto"
else
    echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
fi

#################################################################################

CMD="python $CURR_DIR/src/main.py"
if ! [ -d $CURR_DIR/stores  ]; then mkdir $CURR_DIR/stores; fi
# clean
[[ ! $WORK_DIR =~ ^/ ]] && WORK_DIR=${CURR_DIR}/$WORK_DIR
[ ! -e $WORK_DIR ] && mkdir $WORK_DIR

rm -fr $WORK_DIR/*

DATADIR="$CURR_DIR/stores/store_$(date +%m%d%H%M%S)"
mkdir $DATADIR

# run first block
if [ ! -z $START ]; then    
    cp $START $WORK_DIR 
    $CMD -t $STIME -d -l -s $WORK_DIR
else
    $CMD -t $STIME -d -s $WORK_DIR
fi

CURR_TIME=$(date +%m%d%H%M%S)
for f in $WORK_DIR/log_*; do
    cp $f $DATADIR/$(basename $f)_${CURR_TIME} 
done
cp $WORK_DIR/dumped_robot $DATADIR/dumped_robot_$CURR_TIME 

# run n blocks
if [ $N_BLOCKS -gt 1 ]; then
    for((n=0;n<$[N_BLOCKS-1];n++)); do
        # run n-th block
        $CMD -t $STIME -d -l -s $WORK_DIR
        CURR_TIME=$(date +%m%d%H%M%S)
        for f in $WORK_DIR/log_*; do
            cp $f $DATADIR/$(basename $f)_${CURR_TIME} 
        done
        cp $WORK_DIR/dumped_robot $DATADIR/dumped_robot_$CURR_TIME 
    done
fi
echo done
