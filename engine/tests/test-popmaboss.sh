#!/bin/sh
#
# test1.sh
#

. `dirname $0`/share.sh

return_code=0

if [ ! -x $POPMABOSS ]
then
    echo $POPMABOSS not found
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
echo "PopMaBoSS test"
rm -rf tmp; mkdir -p tmp
/usr/bin/time -p $LAUNCHER $POPMABOSS -c ../examples/popmaboss/Fork.cfg -o tmp/res_fork ../examples/popmaboss/Fork.bnd > /dev/null

if [ $? != 0 ]; then exit 1; fi

/usr/bin/time -p $LAUNCHER $POPMABOSS -c ../examples/popmaboss/Log_Growth.cfg -o tmp/res_log_growth ../examples/popmaboss/Log_Growth.pbnd >/dev/null

if [ $? != 0 ]; then exit 1; fi

/usr/bin/time -p $LAUNCHER $POPMABOSS -c ../examples/popmaboss/Assymetric.cfg -o tmp/res_assymetric ../examples/popmaboss/Assymetric.pbnd > /dev/null

if [ $? != 0 ]; then exit 1; fi

# python compare_probtrajs.py ensemble/refer/res_probtraj.csv tmp/res_probtraj.csv --exact
# check_file "projtraj"
# python compare_probtrajs.py ensemble/refer/res_model_0_probtraj.csv tmp/res_model_0_probtraj.csv --exact
# check_file "projtraj_model_0"
# python compare_probtrajs.py ensemble/refer/res_model_1_probtraj.csv tmp/res_model_1_probtraj.csv --exact
# check_file "projtraj_model_1"
# python compare_probtrajs.py ensemble/refer/res_model_2_probtraj.csv tmp/res_model_2_probtraj.csv --exact
# check_file "projtraj_model_2"
# python compare_probtrajs.py ensemble/refer/res_model_3_probtraj.csv tmp/res_model_3_probtraj.csv --exact
# check_file "projtraj_model_3"
# python compare_probtrajs.py ensemble/refer/res_model_4_probtraj.csv tmp/res_model_4_probtraj.csv --exact
# check_file "projtraj_model_4"
# python compare_probtrajs.py ensemble/refer/res_model_5_probtraj.csv tmp/res_model_5_probtraj.csv --exact
# check_file "projtraj_model_5"

# rm -rf tmp; 

exit $return_code
