#!/bin/bash

set -e

# first of all detect if we are executing ithin the right folder
CURR_DIR=$(pwd)
[ ! -f src/model/Simulation.py ] && echo "you must execute within the project folder" && exit



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
   -t --stime           number of timesteps of a single simulation block
   -g --graph           graphics on
   -w --wdir            working directory
   -S --seed            seed
   -s --start           dumped_robot to start from
   -D --safe-storage    maintain only the last 6 blocks
   -c --clean           clean dumped start file after copy
   -n --n_blocks        number of simulation blocks
   -h --help            show this help

EOF
}



WORK_DIR=
SEED=
START=
STIME=100000
N_BLOCKS=1
GRAPH=false
CLEAN=false
SAFE_STORAGE=false

# getopt
GOTEMP="$(getopt -o "t:w:cn:S:s:Dgh" -l "stime:,wdir:,clean,n_blocks:,seed:,start:,safe-storage,graph,help"  -n '' -- "$@")"


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
        -c | --clean)
            CLEAN=true
            shift;;
        -n | --n_blocks)
            N_BLOCKS="$2"
            shift 2;;
        -S | --seed)
            SEED="$2"
            shift 2;;
        -s | --start)
            START="$2"
            shift 2;;
        -D | --safe-storage)
            SAFE_STORAGE=true
            shift;;
        -g | --graph)
            GRAPH=true
            shift;;
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
ROOT_CMD="python $CURR_DIR/src/main.py"

# run  n-th blocks
for((n=0;n<N_BLOCKS;n++));
do

    snum="$(printf "%06d" $n)"

    GR_OPT=;[[ $GRAPH == true ]] && GR_OPT="-g"
    [[ ! -z $SEED ]] && SEED_OPT="-S $SEED"
    CMD="$ROOT_CMD $GR_OPT $SEED_OPT"
    
    # run first block
    if [[ $n -eq 0 ]]; then

        if [[ -f "$START" ]]; then  # THERE IS a previous dump from which to start
			
            if [[ "$CLEAN" == true ]]; then
                mv $START $WORK_DIR/dumped_robot
            else
                cp $START $WORK_DIR/dumped_robot
            fi

            CMD="$CMD -t $STIME -d -l -s $(pwd)/$WORK_DIR"

        else    # NO previous dumps from which to start

            CMD="$CMD -t $STIME -d -s $(pwd)/$WORK_DIR"

        fi

    else

        # run the following n-th block
        CMD="$CMD -t $STIME -d -l -s $(pwd)/$WORK_DIR"

    fi
    echo -e "\ncommand:\n$CMD\n\n"
    eval "$CMD"

    # store block
    tag=$(date +%Y%m%d%H%M%S)
    for f in $WORK_DIR/*;
    do
        [[ $SAFE_STORAGE == true ]] && rm -fr $(find store|grep dump|sort|head -n -6)
        cp $f store/"$(basename $f)-$tag"
    done

done
echo done
