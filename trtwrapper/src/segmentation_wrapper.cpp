#include "segmentation_wrapper.h"
#include "inference_handlers.h"

SegmentationWrapper::SegmentationWrapper() {
  inference_handler_ = std::make_unique<SegmentationInferenceHandler>();
}

bool SegmentationWrapper::loadFromCudaEngine(const std::string &filename) {
  return inference_handler_->loadFromCudaEngine(filename);
}

bool SegmentationWrapper::inference(cv::Mat &img) {
  std::string status = inference_handler_->inference({img});
  // TODO: Refactor bad legacy
  return status == "OK";
}

std::string SegmentationWrapper::getLastError() {
  return inference_handler_->getLastError();
}

cv::Mat SegmentationWrapper::getColorMask(float alpha) {
  inference_handler_->makeColorMask(alpha);
  return inference_handler_->getColorMask();
}

cv::Mat SegmentationWrapper::getIndexMask() {
  inference_handler_->makeIndexMask();
  return inference_handler_->getIndexMask();
}
