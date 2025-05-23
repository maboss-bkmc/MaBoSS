#!/bin/sh
#
# test-observed_graph.sh
#

. `dirname $0`/share.sh

return_code=0

if [ ! -x $MABOSS_128n ]
then
    echo $MABOSS_128n not found
    exit 1
fi

check_file()
{
    # echo
    if [ $? = 0 ]; then
	echo "File $1 OK"
    else
	echo "File $1 ** error: differences found **"
    return_code=1
    fi
}

echo
echo "Test: Sizek observed graph one thread"
rm -rf tmp; mkdir -p tmp
if [ "$MULTI_THREAD_ONLY" = "" ]; then
    $LAUNCHER $MABOSS_128n sizek/sizek.bnd -c sizek/sizek_wgraph.cfg -e thread_count=1 -o tmp/sizek_1_thread $EXTRA_ARGS
    if [ $? != 0 ]; then exit 1; fi

    diff sizek/refer/sizek_1_thread_observed_graph.csv tmp/sizek_1_thread_observed_graph.csv
    check_file "observed graph"
    
    diff sizek/refer/sizek_1_thread_observed_durations.csv tmp/sizek_1_thread_observed_durations.csv
    check_file "observed durations"
fi

if [ "$ONE_THREAD_ONLY" != "" ]; then exit 0; fi
$LAUNCHER $MABOSS_128n sizek/sizek.bnd -c sizek/sizek_wgraph.cfg -e thread_count=6 -o tmp/sizek_6_thread $EXTRA_ARGS
if [ $? != 0 ]; then exit 1; fi

diff sizek/refer/sizek_6_thread_observed_graph.csv tmp/sizek_6_thread_observed_graph.csv
check_file "observed graph"

diff sizek/refer/sizek_6_thread_observed_durations.csv tmp/sizek_6_thread_observed_durations.csv
check_file "observed durations"

rm -rf tmp; 

exit $return_code
