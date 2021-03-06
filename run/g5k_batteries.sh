#!/bin/bash
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

#!/bin/bash

set -e
IFS=$'\n'

# Manage arguments
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

usage()
{
cat << EOF

usage: $0 options

This script runs batteries of robot simulations  

OPTIONS:
   -t --stime       number of timesteps of a single simulation block
   -n --num         number of simulations
   -g --graph       graphics on
   -s --start       start index of simulation
   -S --seed        seed
   -d --template    template folder containing the exec environment
   -b --dumped      start from a dumped file
   -p --params      set initial parameters interactivelly
   -P --paramfile   set initial parameters from file
   -h --help        show this help

EOF
}

CURR_DIR=$(pwd)
TEMPLATE=${HOME}/working/sensorimotor-development/simulation
TIMESTEPS=20000
ITER=1
START=0
SEED=
DUMPED=false
PARAMS=false
PARAMFILE=
GRAPH=false

# getopt
GOTEMP="$(getopt -o "t:n:gS:s:d:bpP:h" -l "stime:,num:,graph,seed:,start:,template:,dumped,params,paramfile,help"  -n '' -- "$@")"

if ! [ "$(echo -n $GOTEMP |sed -e"s/\-\-.*$//")" ]; then
    usage; exit;
fi

eval set -- "$GOTEMP"

while true ;
do
    case "$1" in
        -t | --stime) 
            TIMESTEPS="$2"
            shift 2;;
        -n | --num) 
            ITER="$2"
            shift 2;;
        -g | --graph) 
            GRAPH=true
            shift;;
        -S | --seed) 
            SEED="$2"
            shift 2;;        
        -s | --start) 
            START="$2"
            shift 2;;
        -d | --template) 
            TEMPLATE="$2"
            shift 2;;
        -b | --dumped) 
            DUMPED=true
            shift;;
        -p | --params) 
            PARAMS=true
            shift;;
        -P | --paramfile) 
            PARAMFILE="$2"
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

cd ${HOME}

# prepare working folder
MAIN_DIR=$(echo $TEMPLATE | sed -e"s/\/simulation//")
TMP_TEMPLATE=/tmp/tmpl_$(basename $TEMPLATE)_$(date +%Y%m%d%H%M%S)
echo "reading data from $MAIN_DIR ..."
cp -r $TEMPLATE $TMP_TEMPLATE
mkdir -p $TMP_TEMPLATE/store
mkdir -p $TMP_TEMPLATE/test
TEMPLATE=$TMP_TEMPLATE


# prepare parameters
if [[ "$PARAMS" == true ]] && [[ -z "$PARAMFILE" ]]; then
    vim $TEMPLATE/src/model/parameters.py
    echo "done parameter setting"
elif [ ! -z "$PARAMFILE" ]; then 
    if [ ! -e ${CURR_DIR}/$PARAMFILE ]; then
        echo $PARAMFILE
        
        echo "must give a file with parameters"
        exit 1
    fi
    for line in $(cat ${CURR_DIR}/$PARAMFILE); do
        KEY=$(echo $line | sed -e"s/^\s*\(.*\+\)\s\+=\s*\(.*\)\s*$/\1/")
        VALUE=$(echo $line | sed -e"s/^\s*\(.*\+\)\s\+=\s*\(.*\)\s*$/\2/")
        
        perl -pi -e "s/${KEY}\s*=.*$/${KEY} = ${VALUE}/" $TEMPLATE/src/model/parameters.py
    done
fi

# :param $1 name of simulation 
run()
{
    local sim_dir=$1
		
    cd ${CURR_DIR}
   	
    DUMP_OPT= 
    if [[ -d "$sim_dir" ]]; then 
        # manage the presence of previous data 
        if [[ "$DUMPED" == false ]]; then 
        	if [[ "$(find $sim_dir| grep dumped | wc -l)" -eq "$ITER" ]]; then
            	# data are complete and we do not want to accumulate new data
            	echo "simulation already completed" 
            	return 0
            fi
        elif [[ "$DUMPED" == true ]]; then
            # we want to continue from previous dumping and accumulate
            DUMPED_FILE="$(find ${sim_dir}/store/|grep dumped_ |sort| tail -n 1)"   
            if [[ -f "$DUMPED_FILE" ]]; then
                DUMP_OPT="-s $(realpath "$DUMPED_FILE")"
                echo "starting from $DUMPED_FILE"
            fi
        fi
    elif [[ ! -d "$sim_dir" ]]; then
        # there are no previous data, create from template
        cp -r $TEMPLATE $sim_dir
        echo "populating $sim_dir"
    fi
    cp $TEMPLATE/src/model/parameters.py $sim_dir/src/model/parameters.py

    cd $sim_dir

    local wdir=test
    echo "starting the simulation..."

    CUM_OPT="-n $ITER"   
    GR_OPT=;[[ "$GRAPH" == true ]] && GR_OPT="-g"
    [[ ! -z "$SEED" ]] && SEED_OPT="-S $SEED"
    
    MAIN_CMD="${MAIN_DIR}/run/run_batch.sh -t $TIMESTEPS $GR_OPT $SEED_OPT -w $wdir $CUM_OPT $DUMP_OPT"
    
    eval "$MAIN_CMD"

    echo "simulation ended"    

    cd ${CURR_DIR}
}

echo start

#--------------------------------------------------
echo "Simulating...."

cd ${CURR_DIR}

name=main_data

run $name &> log_$name 

echo "all done"

rm -fr $TEMPLATE
#--------------------------------------------------
