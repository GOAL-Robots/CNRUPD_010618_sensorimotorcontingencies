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

run()
{
    CURR=$1
    NUM=$2
    curr=$( echo $CURR| sed -e"s/\(.*\)/\L\1\E/")

    cd ${HOME}/working/sensorimotor-development
    
    if [ -z ${curr}-${NUM}.tar.gz ]; then 
        perl -pi -e "s/^(\s*)([^#]+)( # MIXED)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # PRED)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MATCH)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MATCH-2)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MIXED-2)(\s*)$/\1# \2\3\n/" src/model/Robot.py 

        perl -pi -e "s/^(\s*)# ([^#]+)( # $CURR)(\s*)\n$/\1\2\3\n/" src/model/Robot.py 

        wdir=test-$curr-${NUM}

        run/run_batch.sh -t $TIMESTEPS -w $wdir
        if [ $? -ne 0 ];then  
            for f in  $(find $(cat datadir)); 
            do
                cp -r $f ${wdir}/$(echo $(basename $f) | sed -e"s/_[0-9] $//"); 
            done

            R CMD BATCH rscripts/plot.R 
            if [ -f plot.pdf ]; then
                mv plot.pdf ${wdir}/${curr}.pdf  
                tar czvf ${curr}-${NUM}.tar.gz test/
            fi
        fi
    fi
}

for num in $(seq $ITER); 
do
    run MIXED $num & run MIXED-2 $num & run PRED $num &
    wait
    run MATCH $num & run MATCH-2 $num &
    wait
done

