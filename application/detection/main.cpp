#include <iostream>

#include <opencv2/opencv.hpp>

#include "detection_wrapper.h"

using namespace std;

int main() {
  /// Legacy TensortRT. Just for checking that we didn't break anything
  {
    cv::Mat img = cv::imread(
        "/home/luch/Programming/C++/cpp_trt_inference/test_data/images/14_gray.png");

    DetectionWrapper det_wrapper;
    det_wrapper.prepareForInference(
        608, 608,
        "/home/luch/Programming/C++/cpp_trt_inference/"
        "test_data/yolov4_static_nms_gray.bin",
        "/home/luch/Programming/C++/cpp_trt_inference/"
        "labels_for_remap.csv",
        "input",
        {"nms_num_detections", "nms_boxes", "nms_scores", "nms_classes"},
        {0.3, 0.1, 0.1, 0.2, 0.3});

    std::cerr << "Status of load: " << det_wrapper.getLastError() << std::endl;

    det_wrapper.inference({img}, true);
    det_wrapper.setThresh(0.1);
    std::cerr << "Status of inference: " << det_wrapper.getLastError()
              << std::endl;
    std::cout << "Size:  "
              << det_wrapper.getFramesWithBoundingBoxes({img}).size()
              << std::endl;
    cv::Mat im = det_wrapper.getFramesWithBoundingBoxes({img})[0];
    cv::imwrite("/home/luch/Programming/C++/cpp_trt_inference/1_trt_.png", im);

    // That's how you get info without drawing
    std::vector<std::vector<cv::Rect2f>> boxes = det_wrapper.getBoxes();
    std::vector<std::vector<float>> scores = det_wrapper.getScores();
    std::vector<std::vector<int>> classes = det_wrapper.getClasses();

    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Starting inference TRT..." << std::endl;
    for (int i = 0; i < 10000; ++i) {
      auto t1_1 = std::chrono::high_resolution_clock::now();
      det_wrapper.inference({img});
      boxes = det_wrapper.getBoxes();
      scores = det_wrapper.getScores();
      classes = det_wrapper.getClasses();
      auto t2_1 = std::chrono::high_resolution_clock::now();
      cout << "Inference and postprocessing took: " << (t2_1 - t1_1).count()
           << "\n";
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Inference TensorRT took: " << (duration / 10000)
              << " microseconds" << std::endl;
  }

  return 0;
}
