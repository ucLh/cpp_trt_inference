set(MODULE_NAME segmentationWrapperTests)
set(SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/testSegmentationWrapper.cpp)

find_package(OpenCV REQUIRED)
add_executable(${MODULE_NAME} ${SRC_FILES})

target_link_libraries(${MODULE_NAME} gtest gtest_main TRT_WRAPPER ${OpenCV_LIBS})

target_include_directories(${MODULE_NAME} PUBLIC ${TRT_WRAPPER_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})

set(TEST_DATA_PATH ${CMAKE_BINARY_DIR}/tests/${MODULE_NAME})
file(COPY ${PROJECT_SOURCE_DIR}/test_data/1_trt_color.png DESTINATION ${TEST_DATA_PATH})
file(COPY ${PROJECT_SOURCE_DIR}/test_data/1_trt_index.png DESTINATION ${TEST_DATA_PATH})
file(COPY ${PROJECT_SOURCE_DIR}/test_data/left_2_000000318.jpg DESTINATION ${TEST_DATA_PATH})
configure_file(${PROJECT_SOURCE_DIR}/test_data/config.json ${TEST_DATA_PATH}/config.json COPYONLY)

add_test(NAME ${MODULE_NAME} COMMAND ./${MODULE_NAME})