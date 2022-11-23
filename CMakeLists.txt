cmake_minimum_required(VERSION 3.22)

project(matmul CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(matmul)

file(GLOB source_files *.cpp *.cu *.h *.cuh)

target_sources(matmul PRIVATE ${source_files})
target_link_libraries(matmul PRIVATE cublas)
target_include_directories(matmul PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

