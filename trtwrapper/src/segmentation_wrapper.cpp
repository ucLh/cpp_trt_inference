#include "segmentation_wrapper.h"
#include "trt_segmentation_inferencer.h"

SegmentationWrapper::SegmentationWrapper() {
  m_inference_handler = std::make_unique<TRTSegmentationInferencer>();
}

bool SegmentationWrapper::prepareForInference(const std::string &config_path) {
  return m_inference_handler->prepareForInference(config_path);
}

bool SegmentationWrapper::inference(cv::Mat &img) {
  std::string status = m_inference_handler->inference({img});
  // TODO: Refactor bad legacy
  return status == "OK";
}

std::string SegmentationWrapper::getLastError() {
  return m_inference_handler->getLastError();
}

cv::Mat SegmentationWrapper::getColorMask(float alpha,
                                          const cv::Mat &original_image) {
  m_inference_handler->makeColorMask(alpha, original_image);
  return m_inference_handler->getColorMask();
}

cv::Mat SegmentationWrapper::getIndexMask() {
  m_inference_handler->makeIndexMask();
  return m_inference_handler->getIndexMask();
}

void * SegmentationWrapper::getHostDataBuffer() {
  return m_inference_handler->getHostDataBuffer();
}

std::size_t SegmentationWrapper::getHostDataBufferBytesNum() {
  return m_inference_handler->getHostDataBufferBytesNum();
}
