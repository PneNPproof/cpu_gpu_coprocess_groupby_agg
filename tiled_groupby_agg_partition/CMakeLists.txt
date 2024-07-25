cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

# set(CMAKE_C_COMPILER /usr/bin/gcc-11)
# set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.3/bin/nvcc)
project(tiled_groupby_agg_partition LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")

set(TARGET_NAME ${STATIC_LIB_NAME})

include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(${TARGET_NAME} STATIC
            ${PROJECT_SOURCE_DIR}/src/group_then_partition.cu
            ${PROJECT_SOURCE_DIR}/src/kernel.cu
            ${PROJECT_SOURCE_DIR}/src/util.cu
            ${PROJECT_SOURCE_DIR}/src/par_result.cu)     

set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "80" ARCHIVE_OUTPUT_DIRECTORY ${STATIC_LIB_PATH})

target_compile_options(${TARGET_NAME} PRIVATE 
    --compiler-options=-Wall 
    --compiler-options=-Wextra
    --expt-extended-lambda 
    --expt-relaxed-constexpr 
    -Xcompiler -Wno-subobject-linkage
)
