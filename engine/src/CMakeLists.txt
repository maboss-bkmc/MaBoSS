
include(${CMAKE_CURRENT_LIST_DIR}/displayers/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/parsers/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/rpc/CMakeLists.txt)

file(GLOB_RECURSE src_files "${CMAKE_CURRENT_LIST_DIR}/*.cc")
  list(REMOVE_ITEM src_files
    ${CMAKE_CURRENT_LIST_DIR}/PopMaBoSS.cc 
    ${CMAKE_CURRENT_LIST_DIR}/MaBoSS.cc 
    ${CMAKE_CURRENT_LIST_DIR}/MaBoSS-server.cc 
    ${CMAKE_CURRENT_LIST_DIR}/MaBoSS-client.cc 
  )

  if(MSVC)
    list(REMOVE_ITEM src_files 
      ${CMAKE_CURRENT_LIST_DIR}/MBDynBitset.cc
      ${CMAKE_CURRENT_LIST_DIR}/user_func.cc
    )
  endif()

  if(NOT SEDML)
    list(REMOVE_ITEM src_files 
      ${CMAKE_CURRENT_LIST_DIR}/sedml/XMLPatcher.cc
    )
  endif()