#ifndef TRT_DETECTION_INFERENCER_H
#define TRT_DETECTION_INFERENCER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "trt_cnn_inferencer.h"

///
/// \brief The TRTDetectionInferencer class for Tensorflow Object Detection API
///
class TRTDetectionInferencer : public TRTCNNInferencer {
public:
  TRTDetectionInferencer();

  TRTDetectionInferencer(TRTDetectionInferencer &&that);
  virtual ~TRTDetectionInferencer() = default;

  virtual std::string inference(const std::vector<cv::Mat> &imgs);

#ifdef TRT_DEBUG
  ///
  /// \brief getFramesWithBoundingBoxes Debug visualization high-level method
  /// \param tresh
  /// \return Image with drawn bounding boxes
  ///
  std::vector<cv::Mat> getFramesWithBoundingBoxes(float tresh = 0.0f);
#endif

  float getThresh() const;
  void setThresh(float thresh);

  // Access themm after inference done
  std::vector<std::vector<cv::Rect2f>> getBoxes() const;
  std::vector<std::vector<float>> getScores() const;
  std::vector<std::vector<int>> getClasses() const;

protected:
  bool processOutput(const samplesCommon::BufferManager &buffers);

  size_t m_layout_size = 7;
  float m_thresh = 0.5;

  std::vector<std::vector<cv::Rect2f>> m_boxes;
  std::vector<std::vector<float>> m_scores;
  std::vector<std::vector<int>> m_classes;

#ifdef TRT_DEBUG
  std::vector<cv::Mat> m_bb_frames;
  std::vector<cv::Mat> m_frames;
#endif
};

#endif // TRT_DETECTION_INFERENCER_H
