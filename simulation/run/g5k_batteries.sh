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
   -d --template    template folder containing the exec environment
   -h --help        show this help

EOF
}

CURR_DIR=$(pwd)
TEMPLATE=${HOME}/working/sensorimotor-development/simulation
TIMESTEPS=200000
ITER=1


# getopt
GOTEMP="$(getopt -o "t:n:d:h" -l "stime:,num:,template:,help"  -n '' -- "$@")"

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
        -d | --template) 
            TEMPLATE="$2"
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
# mount working folder if we are within g5k network 
if [ -z "$(mount | grep working)" ]; then 
    [ -d working ] && [ -z "$(ls -A working/)" ] && \
        sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 rennes:/home/fmannella/working ${HOME}/working
fi

run()
{
    local CURR=$1
    local NUM=$(printf "%06d" $2)
    local curr=$( echo $CURR| sed -e"s/\(.*\)/\L\1\E/")
    local sim_dir=${curr}_$NUM

    if [ -d $sim_dir ]; then
        echo "simulation already done" 
    else

        cd ${CURR_DIR}
        cp -r $TEMPLATE $sim_dir
        cd $sim_dir 

        perl -pi -e "s/^(\s*)([^#]+)( # MIXED)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # PRED)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MATCH)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MATCH-2)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MIXED-2)(\s*)$/\1# \2\3\n/" src/model/Robot.py 

        perl -pi -e "s/^(\s*)# ([^#]+)( # $CURR)(\s*)\n$/\1\2\3\n/" src/model/Robot.py 

        local wdir=test
        run/run_batch.sh -t $TIMESTEPS -w $wdir 

        echo plot
        R CMD BATCH plot.R  
        if [ -f plot.pdf ]; then
            mv plot.pdf ${wdir}/${curr}.pdf  
        fi
        cd ${CURR_DIR}
    fi 
}

echo start
for n in $(seq $ITER); 
do
    num=$(printf "%06d" $n)
    echo "iter: $n"
    
    cd ${CURR_DIR}

    run MIXED $n > log_mixed_$num 2>&1 &
    run MIXED-2 $n > log_mixed_2_$num 2>&1 &
    run PRED $n > log_pred_$num 2>&1 &
    wait
    run MATCH $n > log_match_$num 2>&1 &
    run MATCH-2 $n > log_match_2_$num 2>&1 &
    wait
done 
echo "all done"
