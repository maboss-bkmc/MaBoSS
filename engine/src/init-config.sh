#!/bin/sh
#
# init-config.sh
#
# 2011-03-28
#
# Eric Viara for Institut Curie copyright 2011
#

CONFIG_NAME=maboss-config
CONFIG_H=${CONFIG_NAME}.h

if [ $# = 0 -a -r $CONFIG_H ]; then exit 0; fi

DONT_USE_BOOST=1
CXXFLAGS="-std=c++11"

tmpfile=/tmp/${CONFIG_NAME}$$.c

cat > $tmpfile <<EOF
#include <boost/unordered_map.hpp>
#include <stdlib.h>
int main()
{
  return 0;
}
EOF

$CXX -c $tmpfile ${CXXFLAGS} > /dev/null 2>&1

if [[ $? == 0 && -z "${DONT_USE_BOOST}" ]]
then
   echo "// @HAS_UNORDERED_MAP"  >> $CONFIG_H
   echo "#define HAS_UNORDERED_MAP" >> $CONFIG_H
   echo "#define HAS_BOOST_UNORDERED_MAP" >> $CONFIG_H
   echo "#include <boost/unordered_map.hpp>" >> $CONFIG_H
   echo "#define STATE_MAP boost::unordered_map" >> $CONFIG_H
   echo "#define HASH_STRUCT hash" >> $CONFIG_H
else
    cat > $tmpfile <<EOF
#include <unordered_map>
#include <stdlib.h>
int main()
{
  return 0;
}
EOF

    $CXX -c $tmpfile ${CXXFLAGS} > /dev/null 2>&1

    if [ $? = 0 ]
    then
	echo "// @HAS_UNORDERED_MAP"  >> $CONFIG_H
	echo "#define HAS_UNORDERED_MAP" >> $CONFIG_H
	echo "#include <unordered_map>" >> $CONFIG_H
	echo "#define STATE_MAP std::unordered_map" >> $CONFIG_H
	echo "#define HASH_STRUCT hash" >> $CONFIG_H
    else
	cat > $tmpfile <<EOF
#include <tr1/unordered_map>
#include <stdlib.h>
int main()
{
  return 0;
}
EOF
	$CXX -c $tmpfile ${CXXFLAGS} > /dev/null 2>&1
	if [ $? = 0 ]
	then
	    echo "// @HAS_UNORDERED_MAP"  >> $CONFIG_H
	    echo "#define HAS_UNORDERED_MAP" >> $CONFIG_H
	    echo "#include <tr1/unordered_map>" >> $CONFIG_H
	    echo "#define STATE_MAP std::tr1::unordered_map" >> $CONFIG_H
	    echo "#define HASH_STRUCT std::tr1::hash" >> $CONFIG_H
	else
	    echo "//#define HAS_UNORDERED_MAP" >> $CONFIG_H
	    echo "#define STATE_MAP std::map" >> $CONFIG_H
	fi
    fi
fi

rm -f $tmpfile ${CONFIG_NAME}*.o



