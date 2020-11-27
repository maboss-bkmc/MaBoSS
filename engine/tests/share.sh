
if [ -z "$MABOSS" ] ; then MABOSS=../src/MaBoSS; fi
if [ -z "$MABOSS_DN" ] ; then MABOSS_DN=../src/MaBoSS_dn; fi
#if [ -z "$MABOSS_BITSET" ] ; then MABOSS_BITSET=../src/MaBoSS_bitset; fi
if [ -z "$MABOSS_128n" ] ; then MABOSS_128n=../src/MaBoSS_128n; fi
if [ -z "$MABOSS_CLIENT" ] ; then MABOSS_CLIENT=../src/MaBoSS-client; fi
if [ -z "$MABOSS_SERVER" ] ; then MABOSS_SERVER=../src/MaBoSS-server; fi
if [ -z "$MABOSS_SERVER_128n" ] ; then MABOSS_SERVER_128n=../src/MaBoSS_128n-server; fi
export DYLD_LIBRARY_PATH=../src:$DYLD_LIBRARY_PATH

if [[ ! -z "$USE_MABOSS_DYNBITSET" ]]; then
    echo "using $MABOSS_DN"
    MABOSS=${MABOSS_DN}
elif [[ ! -z "$USE_BITSET" ]]; then
    echo "using $MABOSS_BITSET"
    MABOSS=${MABOSS_BITSET}
fi
