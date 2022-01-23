#include "segmentation_wrapper.h"

#include "trt_segmentation_inferencer.h"
#include <utility>

SegmentationWrapper::SegmentationWrapper() {
  m_inference_handler = std::make_unique<TRTSegmentationInferencer>();
}

bool SegmentationWrapper::prepareForInference(const std::string &config_path) {
  return m_inference_handler->prepareForInference(config_path);
}

bool SegmentationWrapper::prepareForInference(
    int height, int width, std::string engine_path, std::string colors_path,
    std::string input_node, std::vector<std::string> output_node) {
  IDataBase::ConfigData config;
  config.input_size.height = height;
  config.input_size.width = width;
  config.engine_path = std::move(engine_path);
  config.colors_path = std::move(colors_path);
  config.input_node = std::move(input_node);
  config.output_nodes = std::move(output_node);
  return m_inference_handler->prepareForInference(config);
}

bool SegmentationWrapper::inference(cv::Mat &img) {
  std::string status = m_inference_handler->inference({img});
  // TODO: Refactor bad legacy
  return status == "OK";
}

std::string SegmentationWrapper::getLastError() const {
  return m_inference_handler->getLastError();
}

cv::Mat SegmentationWrapper::getColorMask(float alpha,
                                          const cv::Mat &original_image,
                                          int pixel_sky_border) {
  std::string status = m_inference_handler->makeColorMask(alpha, original_image,
                                                          pixel_sky_border);
  if (status != "OK") {
    std::cerr << status;
    throw exception();
  }
  return m_inference_handler->getColorMask();
}

cv::Mat SegmentationWrapper::getIndexMask(int pixel_sky_border) {
  std::string status = m_inference_handler->makeIndexMask(pixel_sky_border);
  if (status != "OK") {
    std::cerr << status;
    throw exception();
  }
  return m_inference_handler->getIndexMask();
}

void *SegmentationWrapper::getHostDataBuffer() {
  return m_inference_handler->getHostDataBuffer();
}

std::size_t SegmentationWrapper::getHostDataBufferBytesNum() {
  return m_inference_handler->getHostDataBufferBytesNum();
}
