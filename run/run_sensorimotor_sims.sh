#!/bin/bash

#!/bin/bash

set -e

usage()
{
cat << EOF

usage: $0 options

This script runs batteries of robot simulations  

OPTIONS:
   -t --stime       number of timesteps of a single simulation block
   -n --num         number of simulations
   -h --help        show this help

EOF
}

CURR_DIR=$(pwd)
TIMESTEPS=200000
ITER=1


# getopt
GOTEMP="$(getopt -o "t:n:h" -l "stime:,num:,help"  -n '' -- "$@")"

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
if [ -z "$(mount | grep working)" ]; then 
    [ -z "$(ls -A working/)" ] && \
        sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 rennes:/home/fmannella/working ${HOME}/working
fi

cd ${HOME}/working/sensorimotor-development
rm -fr stores/*
tar czvf /tmp/template.tar.gz run src rscripts stores > /dev/null 2>&1
cd $CURR_DIR
[ -d sensorimotor_data ] && rm -fr sensorimotor_data
mkdir sensorimotor_data


run()
{
    local CURR=$1
    local NUM=$2
    local curr=$( echo $CURR| sed -e"s/\(.*\)/\L\1\E/")
    cd ${CURR_DIR}/sensorimotor_data 
    

    mkdir ${curr}_${NUM}
    cd ${curr}_$NUM
    
    tar xzvf /tmp/template.tar.gz > /dev/null 2>&1
   

    perl -pi -e "s/^(\s*)([^#]+)( # MIXED)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
    perl -pi -e "s/^(\s*)([^#]+)( # PRED)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
    perl -pi -e "s/^(\s*)([^#]+)( # MATCH)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
    perl -pi -e "s/^(\s*)([^#]+)( # MATCH-2)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
    perl -pi -e "s/^(\s*)([^#]+)( # MIXED-2)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
    
    perl -pi -e "s/^(\s*)# ([^#]+)( # $CURR)(\s*)\n$/\1\2\3\n/" src/model/Robot.py 

    local wdir=test
    run/run_batch.sh -t $TIMESTEPS -w $wdir 
    echo plot

    R CMD BATCH rscripts/plot.R  
    if [ -f plot.pdf ]; then
        mv plot.pdf ${wdir}/${curr}.pdf  
        tar czvf ${CURR_DIR}/sensorimotor_data/${curr}-${NUM}.tar.gz ${wdir}
    fi
}

for num in $(seq $ITER); 
do
    run MIXED $num > log_mixed_$num 2>&1 &
    run MIXED-2 $num > log_mixed_2_$num 2>&1 &
    run PRED $num > log_pred_$num 2>&1 &
    wait
    run MATCH $num > log_match_$num 2>&1 &
    run MATCH-2 $num > log_match_2_$num 2>&1 &
    wait
done 
echo "all done"
