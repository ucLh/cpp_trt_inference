cmake_minimum_required(VERSION 3.10)
project(TRT_INFERENCE)
set(CMAKE_PREFIX_PATH ~/.local)

set(CMAKE_CXX_STANDARD 17)
set(PROJECT_NAME TRT_INFERENCE)

add_subdirectory(trtwrapper)
#add_subdirectory(application/embeddings)
#add_subdirectory(application/metrics)
add_subdirectory(application/segmentation)
add_subdirectory(application/detection)


include(cmake/googletest.cmake)
fetch_googletest(
        ${PROJECT_SOURCE_DIR}/cmake
        ${PROJECT_BINARY_DIR}/googletest
)

enable_testing()
add_subdirectory(tests)

