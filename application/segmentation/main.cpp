#include <iostream>

#include <opencv2/opencv.hpp>

#include "segmentation_wrapper.h"

using namespace std;

int main() {
  cv::Mat img = cv::imread("left_2_000000318.jpg");

  SegmentationWrapper seg_wrapper;
  seg_wrapper.loadFromCudaEngine("effnetb0_unet_gray_2grass_iou55_640x1280.bin");
  std::cerr << "Status of load: " << seg_wrapper.getLastError() << std::endl;

  seg_wrapper.inference({img});
  std::cerr << "Status of inference: " << seg_wrapper.getLastError()
            << std::endl;
  cv::imwrite("1_trt_index.jpg", seg_wrapper.getIndexMask());
  cv::imwrite("1_trt_color.jpg", seg_wrapper.getColorMask(0.4));
  cv::waitKey(0);

  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Starting inference TRT..." << std::endl;
  for (int i = 0; i < 1000; ++i) {
    seg_wrapper.inference({img});
    seg_wrapper.getIndexMask();
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Inference TensorRT took: " << (duration / 10000)
            << " microseconds" << std::endl;

  return 0;
}

