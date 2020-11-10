#ifndef TRT_INFERENCE_INTERFACES_H
#define TRT_INFERENCE_INTERFACES_H

#include <opencv2/opencv.hpp>
#include <string>

class ISegmentationInferenceHandler {
public:
  virtual std::string inference(const std::vector<cv::Mat> &imgs) = 0;

  virtual std::string makeIndexMask() = 0;

  virtual std::string makeColorMask(float alpha) = 0;

  virtual cv::Mat& getIndexMask() = 0;

  virtual cv::Mat& getColorMask() = 0;

  virtual bool loadFromCudaEngine(const std::string &filename) = 0;

  virtual std::string getLastError() = 0;
};

#endif // TRT_INFERENCE_INTERFACES_H
