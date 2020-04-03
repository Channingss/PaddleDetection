cmake_minimum_required(VERSION 3.0)
project(yml_parse CXX)
include(yaml-cpp.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -o2 -std=c++11")
set(CMAKE_STATIC_LIBRARY_PREFIX "")
include_directories("${CMAKE_SOURCE_DIR}/")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/src/ext-yaml-cpp/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/lib")
link_directories("${CMAKE_CURRENT_BINARY_DIR}")

set(EXTERNAL_LIB "-lrt -ldl -lpthread")
set(DEPS
    yaml-cpp
    ${EXTERNAL_LIB})
add_executable(yml_parse yml_parse.cc)

ADD_DEPENDENCIES(yml_parse ext-yaml-cpp)
target_link_libraries(yml_parse ${DEPS})
