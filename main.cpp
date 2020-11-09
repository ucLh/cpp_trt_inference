#include <iostream>

#include <fstream>
#include <setjmp.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "trt_classification_inferencer.h"
#include "trt_detection_inferencer.h"
#include "trt_segmentation_inferencer.h"

using namespace std;

int main() {
  cv::Mat img = cv::imread("left_2_000000318.jpg");

  TRTSegmentationInferencer inferencer;
  inferencer.original_image = img;
  inferencer.loadFromCudaEngine("effnetb0_unet_gray_2grass_iou55_640x1280.bin");
  std::cerr << "Status of load: " << inferencer.getLastError() << std::endl;

  inferencer.inference({img});
  inferencer.getIndexed();
  inferencer.getColored();
  std::cerr << "Status of inference: " << inferencer.getLastError()
            << std::endl;
  //    std::cout << "Size:  " << inferencer.getFramesWithBoundingBoxes().size()
  //              << std::endl;
  //    cv::Mat im = inferencer.getFramesWithBoundingBoxes()[0];
  cv::imwrite("1_trt_index.jpg", inferencer.getIndexMask());
  cv::imwrite("1_trt_color.jpg", inferencer.getColorMask());
  cv::waitKey(0);

  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Starting inference TRT..." << std::endl;
  for (int i = 0; i < 1000; ++i) {
    inferencer.inference({img});
    inferencer.getIndexed();
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Inference TensorRT took: " << (duration / 10000)
            << " microseconds" << std::endl;

  return 0;
}
