
if [ -z "$MABOSS" ] ; then MABOSS=../src/MaBoSS; fi
if [ -z "$MABOSS_128n" ] ; then MABOSS_128n=../src/MaBoSS_128n; fi
if [ -z "$MABOSS_CLIENT" ] ; then MABOSS_CLIENT=../src/MaBoSS-client; fi
if [ -z "$MABOSS_SERVER" ] ; then MABOSS_SERVER=../src/MaBoSS-server; fi
if [ -z "$MABOSS_SERVER_128n" ] ; then MABOSS_SERVER_128n=../src/MaBoSS_128n-server; fi
export DYLD_LIBRARY_PATH=../src:$DYLD_LIBRARY_PATH
