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

This script builds the online plots  

OPTIONS:
   -d --dir         where to find data
   -v --visualize   visualize on html
   -l --loop        run recursivelly to follow online course
   -h --help        show this help

EOF
}

CURR=$(pwd)
DIR=
LOOP=false
VISUALIZE=false

# getopt
GOTEMP="$(getopt -o "d:lvh" -l "dir:,loop,visualize,help"  -n '' -- "$@")"

if ! [ "$(echo -n $GOTEMP |sed -e"s/\-\-.*$//")" ]; then
    usage; exit;
fi

eval set -- "$GOTEMP"

while true ;
do
    case "$1" in
        -d | --dir) 
            DIR="$2"
            shift 2;;
        -v | --visualize) 
            VISUALIZE=true
            shift;;
        -l | --loop) 
            LOOP=true
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

if [ -z "$DIR" ]; then
    usage; exit;
fi

BASE=$(echo $0|sed -e"s/\/run\/$(basename $0)//")

CURR=$(pwd)
TMP_DIR=/tmp/$(basename $DIR)

[ ! -d "$TMP_DIR" ] && mkdir $TMP_DIR
rm -fr $TMP_DIR/*

echo "data dir: $DIR"
echo "source dir: $BASE"

if [ $VISUALIZE == true ]; then

cat << EOF > $TMP_DIR/plots.html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title> $(basename $DIR) </title>
</head>

<body>
<h1>$(basename $DIR| sed -e"s/sm_//; s/c\([^i]\+\)i/competence \1i/; s/i\(.*\)/ - incompetence \1/; s/\([0-9]\)p\([0-9]\)/\1.\2/g ")</h1>
<table style="width:100%">
  <tr>
    <td><img src="means.png"  width="100%"></td>
    <td><img src="g_means.png"  width="100%"></td>
  </tr>
  <tr>
    <td><img src="gs_means1.png"   width="100%"></td>
    <td><img src="touches.png"  width="100%"></td>
  </tr>
</table>

</body>

</html>
EOF

x-www-browser $TMP_DIR/plots.html & 

fi


run()
{
    cd $TMP_DIR
    
    cat $(find $DIR |grep -v test| grep cont) > $TMP_DIR/log_cont_sensors
    cat $(find $DIR |grep -v test| grep predictions) | sort -k 1 -n | sed -e "s/^/mixed-2 1 /" > $TMP_DIR/all_predictions
    cat $(find $DIR |grep -v test| grep log_sensors) | sort -k 1 -n | sed -e "s/^/mixed-2 1 /" > $TMP_DIR/all_sensors


    R CMD BATCH ${BASE}/rscripts/analyze_touches.R
    R CMD BATCH ${BASE}/rscripts/analyze_predictions.R 
    R CMD BATCH ${BASE}/rscripts/analyze_sensors.R 

    if [ $VISUALIZE == true ]; then
        for f in *.pdf; do
            convert -density 300 -trim $f -quality 100 $(echo $f|sed -e"s/\.pdf/.png/")
        done
    fi
    sleep 1
}

if [ $LOOP == true ]; then
    for((;;)); do run; done
else
    run
fi
