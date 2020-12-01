MESSAGE(STATUS "Including GTest on Tests build")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_MODULE_PATH}../cmake)

# Configuring test data

file(COPY ${PROJECT_SOURCE_DIR}/test_data/effnetb0_unet_gray_2grass_iou55_640x1280_argmax.bin DESTINATION ${CMAKE_BINARY_DIR}/test_data/)

#message(STATUS ${TEST_DATA})
#message(STATUS ${TEST_DATA_DIRECTORY})

add_subdirectory(segmentationWrapperTests)
add_subdirectory(dataHandlingTests)

#configure_file(${PROJECT_SOURCE_DIR}/test_data/config.json ${CMAKE_BINARY_DIR}/tests/segmentationWrapperTests/config.json COPYONLY)
#configure_file(${PROJECT_SOURCE_DIR}/test_data/classes.csv ${CMAKE_BINARY_DIR}/tests/segmentationWrapperTests/classes.csv COPYONLY)