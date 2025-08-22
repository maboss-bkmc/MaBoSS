set CMAKE_BUILD_PARALLEL_LEVEL=1
set CMAKE_GENERATOR=Ninja
set "CMAKE_GENERATOR_PLATFORM="
set "CMAKE_GENERATOR_TOOLSET="
%PYTHON% -m pip install engine/python -vvv --config-settings=cmake.args=-DCMAKE_C_COMPILER=cl.exe;-DCMAKE_CXX_COMPILER=cl.exe;-DSBML=ON;-DLIBBZ_INCLUDE_DIR=%PREFIX%/Library/include;-DLIBBZ_LIBRARY=%PREFIX%/Library/lib/libbz2.lib;-DLIBXML_INCLUDE_DIR=%PREFIX%/Library/include/libxml2;-DLIBXML_LIBRARY=%PREFIX%/Library/lib/libxml2.lib;-DLIBSBML_INCLUDE_DIR=%PREFIX%/Library/include;-DLIBSBML_LIBRARY=%PREFIX%/Library/lib/libsbml.lib
    
mkdir %PREFIX%\tests
mkdir %PREFIX%\tests\cmaboss

xcopy /s engine\tests\cmaboss\ %PREFIX%\tests\cmaboss

