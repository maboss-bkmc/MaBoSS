echo on

cmake -G"Ninja" -S . -B build ^
    -DCMAKE_INSTALL_PREFIX="%PREFIX%" ^
    -DCMAKE_INSTALL_LIBDIR="%PREFIX%"/lib ^
    -DCMAKE_CXX_COMPILER="%CXX%" ^
    -DCMAKE_C_COMPILER="%CC%" ^
    -DLIBSBML_INCLUDE_DIR=%PREFIX%/include ^
    -DLIBSBML_LIBRARY=%PREFIX%/lib/libsbml.lib ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_POLICY_DEFAULT_CMP0074=NEW ^
    -DSTD_THREAD=1 ^
    -DSBML=1
cmake --build build --parallel 1
cmake --install build --component executables 

cmake -G"Ninja" -S . -B build ^
    -DCMAKE_INSTALL_PREFIX="%PREFIX%" ^
    -DCMAKE_INSTALL_LIBDIR="%PREFIX%"/lib ^
    -DCMAKE_CXX_COMPILER="%CXX%" ^
    -DCMAKE_C_COMPILER="%CC%" ^
    -DLIBSBML_INCLUDE_DIR=%PREFIX%/include ^
    -DLIBSBML_LIBRARY=%PREFIX%/lib/libsbml.lib ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_POLICY_DEFAULT_CMP0074=NEW ^
    -DMAXNODES=128 ^
    -DSTD_THREAD=1 ^
    -DSBML=1
cmake --build build --parallel 1
cmake --install build --component executables 

cmake -G"Ninja" -S . -B build ^
    -DCMAKE_INSTALL_PREFIX="%PREFIX%" ^
    -DCMAKE_INSTALL_LIBDIR="%PREFIX%"/lib ^
    -DCMAKE_CXX_COMPILER="%CXX%" ^
    -DCMAKE_C_COMPILER="%CC%" ^
    -DLIBSBML_INCLUDE_DIR=%PREFIX%/include ^
    -DLIBSBML_LIBRARY=%PREFIX%/lib/libsbml.lib ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_POLICY_DEFAULT_CMP0074=NEW ^
    -DMAXNODES=256 ^
    -DSTD_THREAD=1 ^
    -DSBML=1
cmake --build build --parallel 1
cmake --install build --component executables 

cmake -G"Ninja" -S . -B build ^
    -DCMAKE_INSTALL_PREFIX="%PREFIX%" ^
    -DCMAKE_INSTALL_LIBDIR="%PREFIX%"/lib ^
    -DCMAKE_CXX_COMPILER="%CXX%" ^
    -DCMAKE_C_COMPILER="%CC%" ^
    -DLIBSBML_INCLUDE_DIR=%PREFIX%/include ^
    -DLIBSBML_LIBRARY=%PREFIX%/lib/libsbml.lib ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_POLICY_DEFAULT_CMP0074=NEW ^
    -DMAXNODES=512 ^
    -DSTD_THREAD=1 ^
    -DSBML=1
cmake --build build --parallel 1
cmake --install build --component executables 

cmake -G"Ninja" -S . -B build ^
    -DCMAKE_INSTALL_PREFIX="%PREFIX%" ^
    -DCMAKE_INSTALL_LIBDIR="%PREFIX%"/lib ^
    -DCMAKE_CXX_COMPILER="%CXX%" ^
    -DCMAKE_C_COMPILER="%CC%" ^
    -DLIBSBML_INCLUDE_DIR=%PREFIX%/include ^
    -DLIBSBML_LIBRARY=%PREFIX%/lib/libsbml.lib ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_POLICY_DEFAULT_CMP0074=NEW ^
    -DMAXNODES=1024 ^
    -DSTD_THREAD=1 ^
    -DSBML=1
cmake --build build --parallel 1
cmake --install build --component executables 

mkdir %PREFIX%\tests
xcopy /s engine\tests %PREFIX%\tests
