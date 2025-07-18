cmake_minimum_required(VERSION 3.18)
project(LBP LANGUAGES CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SOURCES
    main.cpp
    src/lbp_seq.cpp
    src/lbp_cuda.cu
)

add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)
