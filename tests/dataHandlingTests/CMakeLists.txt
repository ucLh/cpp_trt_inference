set(MODULE_NAME dataHandlingTests)
set(SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/testDataHandling.cpp)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
add_executable(${MODULE_NAME} ${SRC_FILES})

target_link_libraries(${MODULE_NAME} gtest gtest_main TRT_WRAPPER ${OpenCV_LIBS} ${CUDA_LIBRARIES})

target_include_directories(${MODULE_NAME} PUBLIC ${TRT_WRAPPER_SOURCE_DIR} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(TEST_DATA_PATH ${CMAKE_BINARY_DIR}/tests/${MODULE_NAME})

add_test(NAME ${MODULE_NAME} COMMAND ./${MODULE_NAME})