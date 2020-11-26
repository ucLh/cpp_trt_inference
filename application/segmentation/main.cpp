#include <iostream>
#include <vector>

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

  // Example of getting the raw data from the net. Data type is:
  // 1. int for model with argmax op wrapped in its definition.
  // 2. float for model with fp32 OUTPUT precision (used when we want output
  // in CHW format) without argmax wrap.
  // 3. half_float::half (from include/trtwrapper/trt_half.h) for model with
  // fp16 OUTPUT precision (it is used if we want output in HWC format)
  // the time.
  // You can find examples in trt_segmentation_inferencer.cpp (look for
  // processOutput methods).
  auto *buffer = static_cast<int *>(seg_wrapper.getHostDataBuffer());
  size_t buffer_size = seg_wrapper.getHostDataBufferBytesNum() / sizeof(int);
  for (size_t i = 0; i < buffer_size; ++i) {
    cout << buffer[i];
  }
  cout << '\n';

  // Get index or color mask via corresponding method
  cv::imwrite("1_trt_index.png", seg_wrapper.getIndexMask());
  cv::imwrite("1_trt_color.png", seg_wrapper.getColorMask(0.4, img));
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
