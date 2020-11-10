#ifndef TRT_INFERENCE_SEGMENTATION_WRAPPER_H
#define TRT_INFERENCE_SEGMENTATION_WRAPPER_H

#include "interfaces.h"
#include <memory>

class SegmentationWrapper {
public:
  SegmentationWrapper();

  ~SegmentationWrapper() = default;

  bool loadFromCudaEngine(const std::string &filename);

  bool inference(cv::Mat &img);  // Only batch size 1 for now

  std::string getLastError();

  cv::Mat getIndexMask();

  cv::Mat getColorMask(float alpha);

protected:
  std::unique_ptr<ISegmentationInferenceHandler> inference_handler_;
};

#endif // TRT_INFERENCE_SEGMENTATION_WRAPPER_H
