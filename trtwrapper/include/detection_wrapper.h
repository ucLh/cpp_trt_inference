#ifndef TRT_INFERENCE_DETECTION_WRAPPER_H
#define TRT_INFERENCE_DETECTION_WRAPPER_H

#include "interfaces.h"
#include <memory>

class DetectionWrapper {
public:
  DetectionWrapper();

  ~DetectionWrapper() = default;

  //  bool prepareForInference(const std::string &config_path);

  bool prepareForInference(int height, int width, std::string engine_path,
                           std::string labels_path, std::string input_node,
                           std::vector<std::string> output_nodes);

  bool inference(const std::vector<cv::Mat> &imgs,
                 bool apply_postprocessing = false);

  std::string getLastError() const;

  std::vector<cv::Mat>
  getFramesWithBoundingBoxes(const std::vector<cv::Mat> &imgs);

  std::vector<std::vector<cv::Rect2f>> getBoxes() const;
  std::vector<std::vector<float>> getScores() const;
  std::vector<std::vector<int>> getClasses() const;
  float getThresh() const;
  void setThresh(float thresh);

protected:
  std::unique_ptr<IDetectionInferenceHandler> m_inference_handler;
};

#endif // TRT_INFERENCE_DETECTION_WRAPPER_H
