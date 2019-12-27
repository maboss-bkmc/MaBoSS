
if [ -z "$MABOSS" ] ; then MABOSS=../src/MaBoSS; fi
if [ -z "$MABOSS_CLIENT" ] ; then MABOSS_CLIENT=../src/MaBoSS-client; fi
if [ -z "$MABOSS_SERVER" ] ; then MABOSS_SERVER=../src/MaBoSS-server; fi
export DYLD_LIBRARY_PATH=../src:$DYLD_LIBRARY_PATH
