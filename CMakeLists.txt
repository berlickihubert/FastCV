cmake_minimum_required(VERSION 3.18)
project(cuda_matrix_mul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_subdirectory(src/lib)
add_subdirectory(src/utils)
add_subdirectory(src/external)

add_executable(main src/main.cu)

target_link_libraries(main PRIVATE lib utils CImg X11 jpeg png tiff gif)
