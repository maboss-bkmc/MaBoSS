# Grammars : flex, bison stuff
if (MSVC)
set(WINCOMPAT --wincompat)
endif()

find_package(FLEX REQUIRED)
IF(FLEX_FOUND)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers)
    ADD_CUSTOM_COMMAND(
    COMMAND ${FLEX_EXECUTABLE}
    ARGS ${WINCOMPAT} --outfile=${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/lex.CTBNDL.cc
        ${CMAKE_CURRENT_LIST_DIR}/BooleanGrammar.l
    COMMENT "Generating Boolean Network Lexer"
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/lex.CTBNDL.cc
    )
    
    ADD_CUSTOM_COMMAND(
    COMMAND ${FLEX_EXECUTABLE}
    ARGS ${WINCOMPAT} --outfile=${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/lex.RC.cc
        ${CMAKE_CURRENT_LIST_DIR}/RunConfigGrammar.l
    COMMENT "Generating Config Lexer"
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/lex.RC.cc
    )
ENDIF(FLEX_FOUND)

find_package(BISON REQUIRED)
set(BISON_ctbndl_parser_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/BooleanGrammar.cc)
set(BISON_rc_parser_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/RunConfigGrammar.cc)
if(BISON_FOUND)
    ADD_CUSTOM_COMMAND(
    COMMAND ${BISON_EXECUTABLE}
    ARGS -v ${CMAKE_CURRENT_LIST_DIR}/BooleanGrammar.y
        --output=${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/BooleanGrammar.cc
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/lex.CTBNDL.cc
    COMMENT "Generating BooleanGrammar parser.cpp"
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/BooleanGrammar.cc
    )
    
    ADD_CUSTOM_COMMAND(
    COMMAND ${BISON_EXECUTABLE}
    ARGS -v ${CMAKE_CURRENT_LIST_DIR}/RunConfigGrammar.y
        --output=${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/RunConfigGrammar.cc
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/lex.RC.cc
    COMMENT "Generating Runconfig parser.cpp"
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${MABOSS_LIBNAME}.dir/engine/src/parsers/RunConfigGrammar.cc
    )
ENDIF(BISON_FOUND)