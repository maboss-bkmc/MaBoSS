cmake -G"Ninja" -S . -B build \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR="${PREFIX}"/lib \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBXML_INCLUDE_DIR=${PREFIX}/include/libxml2 \
    -DLIBXML_LIBRARY=${PREFIX}/lib/libxml2${SHLIB_EXT} \
    -DLIBSBML_INCLUDE_DIR=${PREFIX}/include \
    -DLIBSBML_LIBRARY=${PREFIX}/lib/libsbml${SHLIB_EXT} \
    -DBUILD_CLIENT=ON -DBUILD_SERVER=ON \
    -DSBML=1
cmake --build build --parallel 1
cmake --install build --component executables 

cmake -G"Ninja" -S . -B build \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR="${PREFIX}"/lib \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBXML_INCLUDE_DIR=${PREFIX}/include/libxml2 \
    -DLIBXML_LIBRARY=${PREFIX}/lib/libxml2${SHLIB_EXT} \
    -DLIBSBML_INCLUDE_DIR=${PREFIX}/include \
    -DLIBSBML_LIBRARY=${PREFIX}/lib/libsbml${SHLIB_EXT} \
    -DMAXNODES=128 \
    -DBUILD_SERVER=ON \
    -DSBML=1
cmake --build build --parallel 1
cmake --install build --component executables 

cmake -G"Ninja" -S . -B build \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR="${PREFIX}"/lib \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBXML_INCLUDE_DIR=${PREFIX}/include/libxml2 \
    -DLIBXML_LIBRARY=${PREFIX}/lib/libxml2${SHLIB_EXT} \
    -DLIBSBML_INCLUDE_DIR=${PREFIX}/include \
    -DLIBSBML_LIBRARY=${PREFIX}/lib/libsbml${SHLIB_EXT} \
    -DMAXNODES=256 \
    -DBUILD_SERVER=ON \
    -DSBML=1
cmake --build build --parallel 1
cmake --install build --component executables 

cmake -G"Ninja" -S . -B build \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR="${PREFIX}"/lib \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBXML_INCLUDE_DIR=${PREFIX}/include/libxml2 \
    -DLIBXML_LIBRARY=${PREFIX}/lib/libxml2${SHLIB_EXT} \
    -DLIBSBML_INCLUDE_DIR=${PREFIX}/include \
    -DLIBSBML_LIBRARY=${PREFIX}/lib/libsbml${SHLIB_EXT} \
    -DMAXNODES=512 \
    -DBUILD_SERVER=ON \
    -DSBML=1
cmake --build build --parallel 1
cmake --install build --component executables 

cmake -G"Ninja" -S . -B build \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR="${PREFIX}"/lib \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBXML_INCLUDE_DIR=${PREFIX}/include/libxml2 \
    -DLIBXML_LIBRARY=${PREFIX}/lib/libxml2${SHLIB_EXT} \
    -DLIBSBML_INCLUDE_DIR=${PREFIX}/include \
    -DLIBSBML_LIBRARY=${PREFIX}/lib/libsbml${SHLIB_EXT} \
    -DMAXNODES=1024 \
    -DBUILD_SERVER=ON \
    -DSBML=1
cmake --build build --parallel 1
cmake --install build --component executables 

cmake -G"Ninja" -S . -B build \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR="${PREFIX}"/lib \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBXML_INCLUDE_DIR=${PREFIX}/include/libxml2 \
    -DLIBXML_LIBRARY=${PREFIX}/lib/libxml2${SHLIB_EXT} \
    -DLIBSBML_INCLUDE_DIR=${PREFIX}/include \
    -DLIBSBML_LIBRARY=${PREFIX}/lib/libsbml${SHLIB_EXT} \
    -DDYNBITSET=ON -DDYNBITSET_STD_ALLOC=ON \
    -DBUILD_SERVER=ON \
    -DSBML=1
cmake --build build --parallel 1
cmake --install build --component executables 

mkdir -p "${PREFIX}/share/MaBoSS"
#mv doc tutorial examples ${PREFIX}/share/MaBoSS/
mv tools/* ${PREFIX}/bin
mkdir -p ${PREFIX}/tests/maboss
cp -r engine/tests/maboss/* ${PREFIX}/tests/maboss

