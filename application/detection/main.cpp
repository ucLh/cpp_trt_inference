#include <iostream>

#include <opencv2/opencv.hpp>

#include "trt_classification_inferencer.h"
#include "trt_detection_inferencer.h"

using namespace std;

int main() {
  /// Legacy TensortRT. Just for checking that we didn't break anything
  {
    cv::Mat img = cv::imread("1.jpg");

    TRTDetectionInferencer inferencer;
    bool ok = inferencer.loadFromCudaEngine("auto_v4_trt.bin");
    // bool ok = inferencer.loadFromUff("ssd_mobilenet_v1_coco.uff");
    std::cerr << "Status of load: " << inferencer.getLastError() << std::endl;

    inferencer.inference({img});
    std::cerr << "Status of inference: " << inferencer.getLastError()
              << std::endl;
    std::cout << "Size:  " << inferencer.getFramesWithBoundingBoxes().size()
              << std::endl;
    cv::Mat im = inferencer.getFramesWithBoundingBoxes()[0];
    cv::imwrite("1_trt_.jpg", im);
    cv::waitKey(0);

    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Starting inference TRT..." << std::endl;
    for (int i = 0; i < 10000; ++i)
      inferencer.inference({img});

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Inference TensorRT took: " << (duration / 10000)
              << " microseconds" << std::endl;
  }

  return 0;
}
