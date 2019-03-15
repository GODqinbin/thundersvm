cmake_minimum_required(VERSION 2.8.12)

if (CMAKE_VERSION VERSION_LESS "3.0")
    project(thundersvm C CXX)
    set(PROJECT_VERSION 0.1.0)
else ()
    cmake_policy(SET CMP0048 NEW)
    project(thundersvm VERSION 0.1.0 LANGUAGES C CXX)
endif ()
set(USE_CUDA OFF CACHE BOOL "Compile with CUDA")
set(USE_EIGEN ON CACHE BOOL "Compile with Eigen")
set(USE_SIMD OFF CACHE BOOL "Compile with SIMD")
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
#find_package(TBB REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
#find_package(MKL REQUIRED)
#include_directories(${MKL_INCLUDE_DIRS})
#link_directories(${MKL_LIBRARIES})
#target_link_libraries(<module>
#mkl_intel_lp64
#mkl_sequential
#mkl_core
#)
#SET(ICC_COMPILE_FLAGS "-O3 -xMIC-AVX512 -fma -align -finline-functions -mkl")
#SET(ICPC_COMPILE_FLAGS "-std=c11 -O3 -xMIC-AVX512 -fma -align -finline-functions -mkl")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#    set(GCC_COMPILE_FLAGS "-O3 -mfma -finline-functions")
#    set(GCPC_COMPILE_FLAGS "-O3 -mfma -finline-functions")
#    set(VEC_FLAGS "-mavx512f -mavx512pf -mavx512er -mavx512cd")
#    if(USE_SIMD)
#	SET(GCC_COMPILE_FLAGS "${GCC_COMPILE_FLAGS} ${VEC_FLAGS}")
#	SET(GCPC_COMPILE_FLAGS "${GCPC_COMPILE_FLAGS} ${VEC_FLAGS}")
#    endif()
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COMPILE_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCPC_COMPILE_FLAGS}")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    SET(ICC_COMPILE_FLAGS "-O3 -fma -align -finline-functions -mkl")
    SET(ICPC_COMPILE_FLAGS "-std=c11 -O3 -fma -align -finline-functions -mkl")
    SET(VEC_FLAGS "-xMIC-AVX512 -save-temps")
    if (USE_SIMD)
        SET(ICC_COMPILE_FLAGS "${ICC_COMPILE_FLAGS} ${VEC_FLAGS}")
        SET(ICPC_COMPILE_FLAGS "${ICPC_COMPILE_FLAGS} ${VEC_FLAGS}")
    endif ()

    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ICC_COMPILE_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ICPC_COMPILE_FLAGS}")
endif()

if (USE_CUDA)
    message("Compile with CUDA")
    find_package(CUDA REQUIRED QUIET)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
    set(LINK_LIBRARY ${CUDA_cusparse_LIBRARY})

    # Correct error for GCC 5 and CUDA 7.5
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
else ()
    message("Compile without CUDA")
endif ()

if (USE_EIGEN)
    message("using Eigen for acceleration")
    include_directories(${PROJECT_SOURCE_DIR}/eigen)
endif ()

#include_directories("/opt/intel/mkl/include")
#include_directories("/opt/intel/tbb/include")
#link_directories("../numa/usr/lib")
#include_directories("../numa/usr/include")
#find_library(numa numa "../numactl")
#find_library(MKL1 mkl_intel_lp64 "/opt/intel/mkl/lib/intel64")
#find_library(MKL2 mkl_core "/opt/intel/mkl/lib/intel64")
#find_library(MKL3 mkl_gnu_thread "/opt/intel/mkl/lib/intel64")
#set(LINK_LIBRARY ${LINK_LIBRARY} ${MKL1} ${MKL2} ${MKL3})
#message(${LINK_LIBRARY})

if (CMAKE_VERSION VERSION_LESS "3.1")
    add_compile_options("-std=c++11")
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

#add_compile_options("-DELPP_FEATURE_PERFORMANCE_TRACKING -DELPP_PERFORMANCE_MICROSECONDS")
add_compile_options("-DELPP_FEATURE_PERFORMANCE_TRACKING")
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})


set(DATASET_DIR ${PROJECT_SOURCE_DIR}/dataset/)
configure_file(include/thundersvm/config.h.in config.h)

set(PROJECT_LIB_NAME ${PROJECT_NAME})
include_directories(${COMMON_INCLUDES})
add_subdirectory(${PROJECT_SOURCE_DIR}/src/thundersvm)


#set(PROJECT_TEST_NAME ${PROJECT_NAME}-test)
#add_subdirectory(${PROJECT_SOURCE_DIR}/src/test)
#add_custom_target(runtest
#        COMMAND ${PROJECT_TEST_NAME})
