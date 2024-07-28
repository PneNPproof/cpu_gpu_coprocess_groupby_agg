cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
# set(CMAKE_C_COMPILER /usr/bin/gcc-11)
# set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.3/bin/nvcc)
project(test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
set(CMAKE_BUILD_TYPE Release)

set(TARGET_NAME test)


include_directories(${STATIC_LIB_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include)
link_directories(${STATIC_LIB_PATH})
# link_libraries(${STATIC_LIB_NAME})
add_executable(${TARGET_NAME}
            ${PROJECT_SOURCE_DIR}/test.cu
            )     

set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "80" RUNTIME_OUTPUT_DIRECTORY ${EXE_PATH})

target_compile_options(${TARGET_NAME} PRIVATE --compiler-options=-Wall --compiler-options=-Wextra
      --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -Wno-subobject-linkage
)

target_link_libraries(${TARGET_NAME} ${STATIC_LIB_NAME})