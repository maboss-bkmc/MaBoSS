
include(${CMAKE_CURRENT_LIST_DIR}/displayers/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/engines/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/parsers/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/rpc/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/sedml/CMakeLists.txt)

list(APPEND src_files
  ${CMAKE_CURRENT_LIST_DIR}/BooleanNetwork.cc 
  ${CMAKE_CURRENT_LIST_DIR}/BuiltinFunctions.cc
  ${CMAKE_CURRENT_LIST_DIR}/Function.cc 
  ${CMAKE_CURRENT_LIST_DIR}/LogicalExprGen.cc
  ${CMAKE_CURRENT_LIST_DIR}/ObservedGraph.cc
  ${CMAKE_CURRENT_LIST_DIR}/ProbaDist.cc
  ${CMAKE_CURRENT_LIST_DIR}/RandomGenerator.cc
  ${CMAKE_CURRENT_LIST_DIR}/RunConfig.cc
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cc
)

if (NOT MSVC)
  list(APPEND src_files
    ${CMAKE_CURRENT_LIST_DIR}/MBDynBitset.cc
    ${CMAKE_CURRENT_LIST_DIR}/user_func.cc
  )
endif()
