find_package(OpenCV REQUIRED)

add_executable(TRT_WRAPPER_DETECTION main.cpp)

target_link_libraries(TRT_WRAPPER_DETECTION TRT_WRAPPER ${OpenCV_LIBS})
target_include_directories(TRT_WRAPPER_DETECTION PRIVATE ${OpenCV_INCLUDE_DIRS})