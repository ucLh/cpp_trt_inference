#ifndef TRT_INFERENCE_INFERENCE_HANDLERS_H
#define TRT_INFERENCE_INFERENCE_HANDLERS_H

#include "interfaces.h"
#include "trt_segmentation_inferencer.h"

class SegmentationInferenceHandler : public ISegmentationInferenceHandler {
public:
  std::string inference(const std::vector<cv::Mat> &imgs) override {
    segm_.original_image = imgs[0];
    return segm_.inference(imgs);
  }

  std::string makeIndexMask() override {
    return segm_.makeIndexMask();
  }

  std::string makeColorMask(float alpha) override {
    segm_.setMixingCoefficient(alpha);
    return segm_.makeColorMask();
  }

  cv::Mat& getIndexMask() override {
    return segm_.getIndexMask();
  }

  cv::Mat& getColorMask() override {
    return segm_.getColorMask();
  }

  bool loadFromCudaEngine(const std::string &filename) override {
    return segm_.loadFromCudaEngine(filename);
  }

  std::string getLastError() override {
    return segm_.getLastError();
  }

private:
  TRTSegmentationInferencer segm_;
};

#endif // TRT_INFERENCE_INFERENCE_HANDLERS_H
