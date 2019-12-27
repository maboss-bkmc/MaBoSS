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
    if [ $? = 0 ]; then
	echo "File $1 OK"
    else
	echo "File $1 ** error: differences found **"
    return_code=1
    fi
}

echo
echo "Server test through container"
rm -rf tmp; mkdir -p tmp
$LAUNCHER /usr/bin/time -p $MABOSS_CLIENT --host localhost --port 7777 -c cellcycle/cellcycle_runcfg.cfg -c cellcycle/cellcycle_runcfg-thread_1.cfg -o tmp/Cell_cycle_thread_1 cellcycle/cellcycle.bnd 
if [ $? != 0 ]; then exit 1; fi
python compare_probtrajs.py cellcycle/refer/Cell_cycle_thread_1_probtraj.csv tmp/Cell_cycle_thread_1_probtraj.csv --exact
check_file "projtraj"

python compare_statdist.py cellcycle/refer/Cell_cycle_thread_1_statdist.csv tmp/Cell_cycle_thread_1_statdist.csv --exact # || echo '**** error test #1.b (non regression) ****'
check_file "statdist"

rm -rf tmp; 

exit $return_code