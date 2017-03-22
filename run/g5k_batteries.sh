#!/bin/bash

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
   -c --cum         cumulative simulations
   -n --num         number of simulations
   -g --graph       graphics on
   -s --start       start index of simulation
   -d --template    template folder containing the exec environment
   -l --learn       learning type [match, match-2,  pred, mixed, mixed-2, mixed-3, all] 
   -b --dumped      start from a dumped file
   -p --params      set initial parameters interactivelly
   -P --paramfile   set initial parameters from file
   -h --help        show this help

EOF
}

CURR_DIR=$(pwd)
TEMPLATE=${HOME}/working/sensorimotor-development/simulation
TIMESTEPS=200000
ITER=1
START=0
DUMPED=false
PARAMS=false
PARAMFILE=
LEARN=all
CUMULATIVE=false
GRAPH=false

# getopt
GOTEMP="$(getopt -o "t:cn:gs:d:l:bpP:h" -l "stime:,cum,num:,graph,start:,template:,learn:,dumped,params,paramfile,help"  -n '' -- "$@")"

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
        -c | --cum) 
            CUMULATIVE=true
            shift;;
        -n | --num) 
            ITER="$2"
            shift 2;;
        -g | --graph) 
            GRAPH=true
            shift;;
        -s | --start) 
            START="$2"
            shift 2;;
        -d | --template) 
            TEMPLATE="$2"
            shift 2;;
        -l | --learn) 
            LEARN="$2"
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
TMP_TEMPLATE=/tmp/$(basename $TEMPLATE)_$(date +%Y%m%d%H%M%S)
cp -r $TEMPLATE $TMP_TEMPLATE
mkdir $TMP_TEMPLATE/store
mkdir $TMP_TEMPLATE/test
TEMPLATE=$TMP_TEMPLATE

# prepare parameters
if [ $PARAMS == true ] && [ -z "$PARAMFILE" ]; then
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

# :param $1 type of siimulation
# :param $2 number oof simulation 
run()
{
    local CURR=$1
    local NUM=$(printf "%06d" $2)
    local curr=$( echo $CURR| sed -e"s/\(.*\)/\L\1\E/")
    local sim_dir=${curr}_$NUM

    cd ${CURR_DIR}
   
    DUMP_OPT= 
    if [ -d $sim_dir ]; then 
        # manage the presence of previous data 
        if [ ! -z "$(find $sim_dir| grep pdf)" ] && [ $DUMPED == false ]; then
            # data are complete and we do not want to accumulate new data
            echo "simulation already completed" 
            return 0
        elif [ $DUMPED == true ]; then 
            # we want to continue from previous dumping and accumulate
            DUMPED_FILE="$(find ${sim_dir}/store/|grep dumped_ |sort| tail -n 1)" 
            DUMP_OPT="-s $(pwd)/$DUMPED_FILE"
            echo "starting from $DUMPED_FILE"  
        fi
    else
        # there are no previous data, create from template
        cp -r $TEMPLATE $sim_dir
        echo "populating $sim_dir"
    fi

    cd $sim_dir
    perl -pi -e "s/^(\s*)([^#]+)( # MIXED)(\s*)$/\1# \2\3\n/" src/model/Simulation.py 
    perl -pi -e "s/^(\s*)([^#]+)( # PRED)(\s*)$/\1# \2\3\n/" src/model/Simulation.py 
    perl -pi -e "s/^(\s*)([^#]+)( # MATCH)(\s*)$/\1# \2\3\n/" src/model/Simulation.py 
    perl -pi -e "s/^(\s*)([^#]+)( # MATCH-2)(\s*)$/\1# \2\3\n/" src/model/Simulation.py 
    perl -pi -e "s/^(\s*)([^#]+)( # MIXED-2)(\s*)$/\1# \2\3\n/" src/model/Simulation.py 
    perl -pi -e "s/^(\s*)([^#]+)( # MIXED-3)(\s*)$/\1# \2\3\n/" src/model/Simulation.py 

    perl -pi -e "s/^(\s*)# ([^#]+)( # $CURR)(\s*)\n$/\1\2\3\n/" src/model/Simulation.py 

    local wdir=test
    echo "starting the simulation..."

    CUM_OPT=; [ $CUMULATIVE == true ] &&  CUM_OPT="-n $ITER"   
    GR_OPT=;[ $GRAPH == true ] && GR_OPT="-g"
    MAIN_CMD="${MAIN_DIR}/run/run_batch.sh -t $TIMESTEPS $GR_OPT -w $wdir $CUM_OPT $DUMP_OPT"
    
    eval "$MAIN_CMD"

    echo "simulation ended"    

    echo "plotting ..."
    R CMD BATCH plot.R  
    if [ -f plot.pdf ]; then
        mv plot.pdf ${wdir}/${curr}.pdf
        echo "plotting ended"  
    else
        echo "plotting failed"
        [ -f plot.Rout ] && cat plot.Rout
    fi

    cd ${CURR_DIR}
}

echo start

if [ $CUMULATIVE == false ]; then
    iterations=$ITER
else
    iterations=1
fi

for n in $(seq $iterations); 
do
    nn=$[n + START - 1]
    num=$(printf "%06d" $nn)
    echo "iter: $nn"
    
    cd ${CURR_DIR}

    cnum=$num
    in=$nn
    
    [ $LEARN == mixed    ] || [ $LEARN == all  ] &&  run MIXED $n &> log_mixed_$num &
    [ $LEARN == mixed-2  ] || [ $LEARN == all  ] &&  run MIXED-2 $n &> log_mixed_2_$num &
    [ $LEARN == pred     ] || [ $LEARN == all  ] &&  run PRED $n &> log_pred_$num &
    wait
    [ $LEARN == match    ] || [ $LEARN == all  ] &&  run MATCH $n &> log_match_$num &
    [ $LEARN == match-2  ] || [ $LEARN == all  ] &&  run MIXED-3 $n &> log_mixed_3_$num &
    [ $LEARN == match-3  ] || [ $LEARN == all  ] &&  run MATCH-2 $n &> log_match_2_$num &
    wait
    sleep 1
done 
echo "all done"

rm -fr $TEMPLATE 
