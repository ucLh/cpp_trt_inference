#include "detection_wrapper.h"
#include "detection_inference/trt_effdet_inferencer.h"
#include "detection_inference/trt_yolo_inferencer.h"

DetectionWrapper::DetectionWrapper(DetectionInferencerType type) {
  if (type == DetectionInferencerType::YOLO) {
    m_inference_handler = std::make_unique<TRTYoloInferencer>();
  }
  else if (type == DetectionInferencerType::EFFDET) {
    m_inference_handler = std::make_unique<TRTEffdetInferencer>();
  }
}

bool DetectionWrapper::prepareForInference(
    int height, int width, std::string engine_path, std::string labels_path,
    std::string input_node, std::vector<std::string> output_node,
    bool show_object_class, std::vector<float> categories_thresholds) {
  IDataBase::ConfigData config;
  config.input_size.height = height;
  config.input_size.width = width;
  config.engine_path = std::move(engine_path);
  config.detection_labels_path = std::move(labels_path);
  config.input_node = std::move(input_node);
  config.output_nodes = std::move(output_node);
  config.categories_thresholds = std::move(categories_thresholds);
  config.show_object_class = show_object_class;
  return m_inference_handler->prepareForInference(config);
}

bool DetectionWrapper::inference(const std::vector<cv::Mat> &imgs) {
  std::string status = m_inference_handler->inference(imgs);
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
