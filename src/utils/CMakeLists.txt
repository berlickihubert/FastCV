# CMakeLists for utils library
add_library(utils STATIC utils.cpp)

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
