#!/bin/bash

#!/bin/bash

set -e

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
   -s --start       start i dex of simulation
   -d --template    template folder containing the exec environment
   -l --learn       learning type [match, match-2,  pred, mixed, mixed-2, mixed-3, all] 
   -b --dumped      start from a dumped file
   -p --params      set initial parameters interactivelly
   -h --help        show this help

EOF
}

CURR_DIR=$(pwd)
TEMPLATE=${HOME}/working/sensorimotor-development/simulation
TIMESTEPS=200000
ITER=0
START=0
DUMPED=false
PARAMS=false
LEARN=all

# getopt
GOTEMP="$(getopt -o "t:n:s:d:l:bph" -l "stime:,num:,start:,template:,learn:,dumped,params,help"  -n '' -- "$@")"

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
# mount working folder if we are within g5k network 
if [ -z "$(mount | grep working)" ]; then 
    [ -d working ] && [ -z "$(ls -A working/)" ] && \
        sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 rennes:/home/fmannella/working ${HOME}/working
fi

# parameters
if [ $PARAMS == true ]; then
    TMP_TEMPLATE=/tmp/$(basename $TEMPLATE)_$(date +%Y%m%d%H%M%S)
    cp -r $TEMPLATE $TMP_TEMPLATE
    TEMPLATE=$TMP_TEMPLATE
    vim $TEMPLATE/src/model/parameters.py
    echo "done parameter setting"
fi

run()
{
    local CURR=$1
    local NUM=$(printf "%06d" $2)
    local curr=$( echo $CURR| sed -e"s/\(.*\)/\L\1\E/")
    local sim_dir=${curr}_$NUM

    if [ -d $sim_dir ] && \
        [ ! -z "$(find $sim_dir| grep pdf)" ] && \
        [ $DUMPED == false ]; then
        echo "simulation already completed" 
    else

        cd ${CURR_DIR}
        cp -r $TEMPLATE $sim_dir
        cd $sim_dir 

        perl -pi -e "s/^(\s*)([^#]+)( # MIXED)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # PRED)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MATCH)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MATCH-2)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MIXED-2)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MIXED-3)(\s*)$/\1# \2\3\n/" src/model/Robot.py 

        perl -pi -e "s/^(\s*)# ([^#]+)( # $CURR)(\s*)\n$/\1\2\3\n/" src/model/Robot.py 

        local wdir=test
        echo "starting the simulation..."

        run/run_batch.sh -t $TIMESTEPS -w $wdir $([ $DUMPED == true ] && echo -n "-s $wdir/dumped_robot" )
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
    fi 
}

echo start
for n in $(seq $ITER); 
do
    nn=$[n + $START]
    num=$(printf "%06d" $nn)
    echo "iter: $nn"
    
    cd ${CURR_DIR}

    [ $LEARN == mixed    ] || [ $LEARN == all  ] &&  run MIXED $n > log_mixed_$num 2>&1 &
    [ $LEARN == mixed-2  ] || [ $LEARN == all  ] &&  run MIXED-2 $n > log_mixed_2_$num 2>&1 &
    [ $LEARN == pred     ] || [ $LEARN == all  ] &&  run PRED $n > log_pred_$num 2>&1 &
    wait
    [ $LEARN == match    ] || [ $LEARN == all  ] &&  run MATCH $n > log_match_$num 2>&1 &
    [ $LEARN == match-2  ] || [ $LEARN == all  ] &&  run MIXED-3 $n > log_mixed_3_$num 2>&1 &
    [ $LEARN == match-3  ] || [ $LEARN == all  ] &&  run MATCH-2 $n > log_match_2_$num 2>&1 &
    wait
done 
echo "all done"
