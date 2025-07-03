add_library(lib STATIC matrix_mul.cu sift.cu)

target_include_directories(lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
