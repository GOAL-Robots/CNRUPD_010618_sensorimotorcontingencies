#!/bin/bash

TIMESTEPS=$1

if [ -z "$(mount | grep working)" ]; then 
    sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 rennes:/home/fmannella/working ${HOME}/working
fi

run()
{
    CURR=$1
    curr=$( echo $CURR| sed -e"s/\(.*\)/\L\1\E/")

    cd ${HOME}/working/sensorimotor-development
    
    if [ -z ${curr}.tar.gz ]; then 
        perl -pi -e "s/^(\s*)([^#]+)( # MIXED)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # PRED)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MATCH)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MATCH-2)(\s*)$/\1# \2\3\n/" src/model/Robot.py 
        perl -pi -e "s/^(\s*)([^#]+)( # MIXED-2)(\s*)$/\1# \2\3\n/" src/model/Robot.py 

        perl -pi -e "s/^(\s*)# ([^#]+)( # $CURR)(\s*)\n$/\1\2\3\n/" src/model/Robot.py 

        run/run_batch.sh -t $TIMESTEPS
        if [ $? -ne 0 ];then  
            for f in  $(find $(cat datadir)); 
            do
                cp -r $f test/$(echo $(basename $f) | sed -e"s/_[0-9] $//"); 
            done

            R CMD BATCH rscripts/plot.R 
            if [ -f plot.pdf ]; then
                mv plot.pdf test/${curr}.pdf  
                tar czvf ${curr}.tar.gz test/
            fi
        fi
    fi
}

# MIXED
run MIXED 
run MIXED-2 

# PRED
run PRED

# MATCH
run MATCH
run MATCH-2

