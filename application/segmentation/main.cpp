#include <iostream>

#include <opencv2/opencv.hpp>

#include "segmentation_wrapper.h"

using namespace std;

int main() {
  cv::Mat img = cv::imread("left_2_000000318.jpg");

  // Create wrapper.
  SegmentationWrapper seg_wrapper;

  // Prepare wrapper for inference. You have to give a path to the config.
  seg_wrapper.prepareForInference("config.json");
  // Check if network loaded.
  std::cerr << "Status of load: " << seg_wrapper.getLastError() << std::endl;

  // Inference an image.
  seg_wrapper.inference(img);
  std::cerr << "Status of inference: " << seg_wrapper.getLastError()
            << std::endl;
  // Get index or color mask via corresponding method
  cv::imwrite("1_trt_index.jpg", seg_wrapper.getIndexMask());
  cv::imwrite("1_trt_color.jpg", seg_wrapper.getColorMask(0.4, img));
  cv::waitKey(0);

  // Time measurement. Inference one picture over and over again.
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Starting inference TRT..." << std::endl;
  for (int i = 0; i < 1000; ++i) {
    auto t1_1 = std::chrono::high_resolution_clock::now();
    seg_wrapper.inference(img);
    seg_wrapper.getIndexMask();
    auto t1_2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t1_2 - t1_1)
            .count();
    std::cout << "Inference + postprocessing took: " << duration << "\n";
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Inference TensorRT took: " << (duration / 1000)
            << " microseconds" << std::endl;

  return 0;
}
