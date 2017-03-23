#!/bin/bash
#!/bin/bash


DIR=$1
BASE=$(echo $0|sed -e"s/run\/$(basename $0)//")

cat $(find $DIR | grep cont) > log_cont_sensors
cat $(find $DIR | grep predictions) | sort -k 1 -n | sed -e "s/^/mixed-2 1 /" > all_predictions
cat $(find $DIR | grep log_sensors) | sort -k 1 -n | sed -e "s/^/mixed-2 1 /" > all_sensors

Rexec ${BASE}/rscripts/analyze_touches.R
Rexec ${BASE}/rscripts/analyze_predictions.R 
Rexec ${BASE}/Dropbox/rscripts/analyze_sensors.R 


