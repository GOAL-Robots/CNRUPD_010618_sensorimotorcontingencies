#!/bin/bash


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
   -g --graph       make graphs   
   -w --www         open browser
   -l --loop        run recursivelly to follow online course
   -h --help        show this help

EOF
}

WWW=false
CURR=$(pwd)
DIR=
LOOP=
VISUALIZE=false
GRAPHS=false

# getopt
GOTEMP="$(getopt -o "d:gwl:h" -l "dir:,graphs,www,loop,help"  -n '' -- "$@")"

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
        -g | --graphs)
            GRAPHS=true
            shift;;
        -w | --www)
            WWW=true
            shift;;
        -l | --loop)
            LOOP=$2
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

if [ -z "$DIR" ]; then
    usage; exit;
fi

BASE=$(echo $0|sed -e"s/\/run\/$(basename $0)//")
BASE=$(readlink -e $BASE)
DIR=$(readlink -e $DIR)

CURR=$(pwd)
TMP_DIR=/tmp/$(basename $DIR)_plots

[ ! -d "$TMP_DIR" ] && mkdir $TMP_DIR
rm -fr $TMP_DIR/*

echo "data dir: $DIR"
echo "source dir: $BASE"

if [ $GRAPHS == true ] && [ $WWW == true ]; then

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
  <tr>
    <td><img src="gs_means_goal1.png"   width="100%"></td>
    <td></td>
  </tr>
</table>

</body>

</html>
EOF
fi

if [ $WWW == true ] && [ $GRAPHS == true ]; then
    x-www-browser $TMP_DIR/plots.html &
    sleep 2
fi

run()
{
    cd $TMP_DIR

    echo "collect data..."
    cat $(find $DIR | grep cont) > $TMP_DIR/log_cont_sensors
    cat $(find $DIR | grep predictions) | sed -e"s/\s\+/ /g; s/[^[:print:]]//g" | sort -k 1 -n | sed -e "s/^/SIM 1 /" > $TMP_DIR/all_predictions
    cat $(find $DIR | grep log_sensors) | sed -e"s/\s\+/ /g; s/[^[:print:]]//g" | sort -k 1 -n | sed -e "s/^/SIM 1 /" > $TMP_DIR/all_sensors

    if [ $GRAPHS == true ]; then
        echo "run R scripts..."
        R CMD BATCH ${BASE}/rscripts/analyze_touches.R
        R CMD BATCH ${BASE}/rscripts/analyze_predictions.R
        R CMD BATCH ${BASE}/rscripts/analyze_sensors.R 

        echo "convert images to png..."
        for f in *.pdf; do
            convert -density 300 -trim $f -quality 100 $(echo $f|sed -e"s/\.pdf/.png/")
        done
        echo "done"
    fi
}

run
if [ ! -z "$LOOP" ]; then
    for((;;)); do run; sleep $LOOP; done
fi
