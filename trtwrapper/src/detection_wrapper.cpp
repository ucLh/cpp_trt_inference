#include "detection_wrapper.h"
#include "trt_detection_inferencer.h"

DetectionWrapper::DetectionWrapper() {
  m_inference_handler = std::make_unique<TRTDetectionInferencer>();
}

bool DetectionWrapper::prepareForInference(
    int height, int width, std::string engine_path, std::string labels_path,
    std::string input_node, std::vector<std::string> output_node) {
  IDataBase::ConfigData config;
  config.input_size.height = height;
  config.input_size.width = width;
  config.engine_path = std::move(engine_path);
  config.detection_labels_path = std::move(labels_path);
  config.input_node = std::move(input_node);
  config.output_nodes = std::move(output_node);
  return m_inference_handler->prepareForInference(config);
}

bool DetectionWrapper::inference(const std::vector<cv::Mat> &imgs,
                                 bool apply_postprocessing) {
  std::string status = m_inference_handler->inference(imgs, apply_postprocessing);
  // TODO: Refactor bad legacy
  return status == "OK";
}

std::string DetectionWrapper::getLastError() const {
  return m_inference_handler->getLastError();
}

std::vector<cv::Mat>
DetectionWrapper::getFramesWithBoundingBoxes(const std::vector<cv::Mat> &imgs) {
  return m_inference_handler->getFramesWithBoundingBoxes(imgs);
}

std::vector<std::vector<cv::Rect2f>> DetectionWrapper::getBoxes() const {
  return m_inference_handler->getBoxes();
}

std::vector<std::vector<float>> DetectionWrapper::getScores() const {
  return m_inference_handler->getScores();
}

std::vector<std::vector<int>> DetectionWrapper::getClasses() const {
  return m_inference_handler->getClasses();
}

float DetectionWrapper::getThresh() const {
  return m_inference_handler->getThresh();
}

void DetectionWrapper::setThresh(float thresh) {
  m_inference_handler->setThresh(thresh);
}
