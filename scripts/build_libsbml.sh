#!/bin/bash

if [ -z "${INSTALL_PATH}" ]
then 
INSTALL_PATH=/usr/local/
fi


if [ -z "${BUILD_PATH}" ]
then 
BUILD_PATH=/tmp/
fi

if [ -z "${CXX}" ]
then 
CXX=g++
fi

if [ -z "${GCC}" ]
then 
GCC=gcc
fi

if [ -z "${LIBXML_LIBDIR}" ]
then 
LIBXML_LIBDIR=/usr/lib
fi

if [ -z "${LIBXML_INCLUDEDIR}" ]
then 
LIBXML_INCLUDEDIR=/usr/include/libxml2
fi

mkdir -p ${BUILD_PATH}
cd ${BUILD_PATH}
wget https://sourceforge.net/projects/sbml/files/libsbml/5.19.0/stable/libSBML-5.19.0-core-plus-packages-src.tar.gz
tar -zxf libSBML-5.19.0-core-plus-packages-src.tar.gz
cd libSBML-5.19.0-Source
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
-DCMAKE_INSTALL_LIBDIR=${INSTALL_PATH}/lib \
-DCMAKE_CXX_COMPILER=${CXX} \
-DCMAKE_C_COMPILER=${GCC} \
-DCMAKE_CXX_STANDARD_LIBRARIES=-lxml2 \
-DWITH_SWIG=OFF \
-DLIBXML_LIBRARY=${LIBXML_LIBDIR} \
-DLIBXML_INCLUDE_DIR=${LIBXML_INCLUDEDIR} \
-DENABLE_COMP=ON \
-DENABLE_FBC=ON \
-DENABLE_GROUPS=ON \
-DENABLE_LAYOUT=ON \
-DENABLE_MULTI=ON \
-DENABLE_QUAL=ON \
-DENABLE_RENDER=ON \
-DENABLE_DISTRIB=ON \
-DWITH_CPP_NAMESPACE=ON \
..
make