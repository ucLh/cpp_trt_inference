#ifndef TRT_INFERENCE_SEGMENTATION_WRAPPER_H
#define TRT_INFERENCE_SEGMENTATION_WRAPPER_H

#include "interfaces.h"
#include <memory>

class SegmentationWrapper {
public:
  SegmentationWrapper();

  ~SegmentationWrapper() = default;

  bool prepareForInference(const std::string &config_path);

  bool inference(cv::Mat &img); // Only batch size 1 for now

  std::string getLastError();

  cv::Mat getIndexMask(int pixel_sky_border = 0);

  cv::Mat getColorMask(float alpha, const cv::Mat &original_image,
                       int pixel_sky_border = 0);

  void *getHostDataBuffer();

  size_t getHostDataBufferBytesNum();

protected:
  std::unique_ptr<ISegmentationInferenceHandler> m_inference_handler;
};

#endif // TRT_INFERENCE_SEGMENTATION_WRAPPER_H
