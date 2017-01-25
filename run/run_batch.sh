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


find_maindir()
{
    local current_dir=$(pwd)
    local parent_dir=$current_dir
    while [ ! -z $parent_dir ]; do 
        if [ -d $parent_dir ]; then
            local parent_children=$(ls $parent_dir) 
            if [[ "$parent_children" =~ MAIN_DIR ]]; then
                echo $parent_dir
                break
            fi
        fi
        local parent_dir=$(echo $parent_dir| sed -e"s/\/[^\/]\+$//"); 
    done
}

MAIN_DIR="$(find_maindir)"

if [ -z $MAIN_DIR ]; then
    echo "you must execute within the project dirrctory"
fi



WORK_DIR=$MAIN_DIR/test

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

    echo "OMP_NUM_THREADSì auto"
else
    echo "OMP_NUM_THREADSì: $OMP_NUM_THREADS"
fi

#################################################################################

CMD="python $MAIN_DIR/src/main.py"
if ! [ -d $MAIN_DIR/stores  ]; then mkdir $MAIN_DIR/stores; fi
DATADIR="$MAIN_DIR/stores/store_$(date +%m%d%H%M%S)"
mkdir $DATADIR



# clean

if [ ! -e $WORK_DIR ]; then
    mkdir $WORK_DIR
fi

rm -fr $WORK_DIR/*

# run first block
if [ ! -z $START ]; then    
    cp $START $WORK_DIR 
    $CMD -t $STIME -d -l -s $WORK_DIR
else
    $CMD -t $STIME -d -s $WORK_DIR
fi

CURR_TIME=$(date +%m%d%H%M%S)
for f in $WORK_DIR/log_*; do
    mv $f $DATADIR/$(basename $f)_${CURR_TIME} 
done
cp $WORK_DIR/dumped_robot $DATADIR/dumped_robot_$CURR_TIME 

# run n blocks
if [ $N_BLOCKS -gt 1 ]; then
    for((n=0;n<$[N_BLOCKS-1];n++)); do
        # run n-th block
        $CMD -t $STIME -d -l -s $WORK_DIR
        CURR_TIME=$(date +%m%d%H%M%S)
        for f in $WORK_DIR/log_*; do
            mv $f $DATADIR/$(basename $f)_${CURR_TIME} 
        done
        cp $WORK_DIR/dumped_robot $DATADIR/dumped_robot_$CURR_TIME 
    done
fi

echo $DATADIR > datadir
