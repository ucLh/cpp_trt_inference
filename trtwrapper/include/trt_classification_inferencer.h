#ifndef TRT_CLASSIFICATION_INFERENCER_H
#define TRT_CLASSIFICATION_INFERENCER_H

#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "trt_cnn_inferencer.h"

///
/// \brief The TRTClassificationInferencer class for Tensorflow Slim
/// Claassification API
///
class TRTClassificationInferencer : public TRTCNNInferencer {
public:
  TRTClassificationInferencer();

  TRTClassificationInferencer(TRTClassificationInferencer &&that);
  virtual ~TRTClassificationInferencer() = default;

  virtual std::string inference(const std::vector<cv::Mat> &imgs);

  float getThresh() const;
  void setThresh(float thresh);

  std::vector<int> getClasses() const;
  std::vector<float> getScores() const;

protected:
  bool processOutput(const samplesCommon::BufferManager &buffers);

  std::vector<float> m_scores;
  std::vector<int> m_classes;

  float m_thresh = 0.5;
};

#endif // TRT_CLASSIFICATION_INFERENCER_H
