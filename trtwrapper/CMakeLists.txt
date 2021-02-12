cmake_minimum_required(VERSION 3.10)
project(TRT_WRAPPER)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

find_package(OpenCV 4 REQUIRED)
find_package(CUDA REQUIRED)

add_library(TRT_WRAPPER
            src/trt_segmentation_inferencer.cpp
            src/trt_classification_inferencer.cpp
            src/trt_cnn_inferencer.cpp
            src/trt_detection_inferencer.cpp
            src/logger.cpp
            src/segmentation_wrapper.cpp
            src/data_handler.cpp
            src/detection_wrapper.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
message("TRT libs:" ${TensorRT_LIBRARIES})

target_include_directories(TRT_WRAPPER PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(TRT_WRAPPER PRIVATE ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(TRT_WRAPPER PRIVATE stdc++fs pthread ${OpenCV_LIBS}
        ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} nvonnxparser nvinfer nvinfer_plugin nvparsers)


