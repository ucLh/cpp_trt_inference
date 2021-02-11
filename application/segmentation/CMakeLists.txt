find_package(OpenCV REQUIRED)

add_executable(TRT_WRAPPER_SEGMENTATION main.cpp)

configure_file(config.json ${CMAKE_BINARY_DIR}/application/segmentation COPYONLY)
file(COPY ${PROJECT_SOURCE_DIR}/test_data/images/left_2_000000318.jpg DESTINATION ${CMAKE_BINARY_DIR}/application/segmentation)

target_link_libraries(TRT_WRAPPER_SEGMENTATION TRT_WRAPPER ${OpenCV_LIBS})
target_include_directories(TRT_WRAPPER_SEGMENTATION PRIVATE ${OpenCV_INCLUDE_DIRS})
#target_include_directories(TRT_WRAPPER_SEGMENTATION_EXAMPLE PUBLIC ${TRT_WRAPPER_SEGMENTATION_SOURCE_DIR})