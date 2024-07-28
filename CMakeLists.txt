cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(tiled_groupby_agg_partition_lib_and_test LANGUAGES CXX CUDA) 

# # specify the C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_VERBOSE_MAKEFILE ON)
add_definitions(-DMEASURE_TIME)

set(STATIC_LIB_PATH ${PROJECT_SOURCE_DIR}/lib)
set(STATIC_LIB_NAME tiled_groupby_agg_partition)
set(EXE_PATH ${PROJECT_SOURCE_DIR}/bin)
set(STATIC_LIB_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/tiled_groupby_agg_partition/include)

add_subdirectory(tiled_groupby_agg_partition)
add_subdirectory(test)