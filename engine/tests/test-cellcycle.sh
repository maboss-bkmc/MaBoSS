#!/bin/sh
#
# test1.sh
#

. `dirname $0`/share.sh

return_code=0

if [ ! -x $MABOSS ]
then
    echo $MABOSS not found
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
echo "Non regression test: Cell Cycle one thread"
rm -rf tmp; mkdir -p tmp
$LAUNCHER /usr/bin/time -p $MABOSS cellcycle/cellcycle.bnd -c cellcycle/cellcycle_runcfg.cfg -c cellcycle/cellcycle_runcfg-thread_1.cfg -o tmp/Cell_cycle_thread_1
if [ $? != 0 ]; then exit 1; fi
python compare_probtrajs.py cellcycle/refer/Cell_cycle_thread_1_probtraj.csv tmp/Cell_cycle_thread_1_probtraj.csv --exact
check_file "projtraj"

python compare_statdist.py cellcycle/refer/Cell_cycle_thread_1_statdist.csv tmp/Cell_cycle_thread_1_statdist.csv --exact # || echo '**** error test #1.b (non regression) ****'
check_file "statdist"

if [ "$ONE_THREAD_ONLY" != "" ]; then exit 0; fi

echo
echo "Non regression test: Cell Cycle 6 threads"
$LAUNCHER /usr/bin/time -p $MABOSS cellcycle/cellcycle.bnd -c cellcycle/cellcycle_runcfg.cfg -c cellcycle/cellcycle_runcfg-thread_6.cfg -o tmp/Cell_cycle_thread_6

python compare_probtrajs.py cellcycle/refer/Cell_cycle_thread_6_probtraj.csv tmp/Cell_cycle_thread_6_probtraj.csv --exact
check_file "projtraj"

python compare_statdist.py cellcycle/refer/Cell_cycle_thread_6_statdist.csv tmp/Cell_cycle_thread_6_statdist.csv --exact #|| echo '**** error test #2.b (non regression) ****'
check_file "statdist"

echo
echo "Non regression test: checking differences between one and 6 threads results"
python compare_probtrajs.py tmp/Cell_cycle_thread_1_probtraj.csv tmp/Cell_cycle_thread_6_probtraj.csv # || echo '**** error test #3 (multi threads) ****'
check_file "probtrajs"
python compare_statdist.py tmp/Cell_cycle_thread_1_statdist.csv tmp/Cell_cycle_thread_6_statdist.csv --exact # || echo '**** error test #3 (multi threads) ****'
check_file "statdist"

exit $return_code