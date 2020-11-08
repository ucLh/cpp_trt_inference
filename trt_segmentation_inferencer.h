#ifndef TRTWRAPPER_PROJ_TRT_SEGMENTATION_H
#define TRTWRAPPER_PROJ_TRT_SEGMENTATION_H

#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "trt_cnn_inferencer.h"

///
/// \brief The TRTClassificationInferencer class for Tensorflow Slim
/// Claassification API
///
class TRTSegmentationInferencer : public TRTCNNInferencer {
public:
  TRTSegmentationInferencer();

  TRTSegmentationInferencer(TRTSegmentationInferencer &&that);
  virtual ~TRTSegmentationInferencer() = default;

  virtual std::string inference(const std::vector<cv::Mat> &imgs);

  float getThresh() const;
  void setThresh(float thresh);

  std::vector<int> getClasses() const;
  std::vector<float> getScores() const;

  int rows = 640;
  int cols = 1280;
  cv::Mat original_image;

protected:
  // bool buffer2Mat(const samplesCommon::BufferManager &buffers);

  bool processOutput(const samplesCommon::BufferManager &buffers, const std::vector<cv::Mat> &imgs);
  bool processOutput(const samplesCommon::BufferManager &buffers);

  std::vector<float> _scores;
  std::vector<int> _classes;
  std::vector<cv::Mat> _probs;
  cv::Mat _colored_mask;

  float _thresh = 0.5;
};

#endif //TRTWRAPPER_PROJ_TRT_SEGMENTATION_H
