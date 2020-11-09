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

  std::string inference(const std::vector<cv::Mat> &imgs) override;
  std::string makeIndexMask();
  std::string makeColorMask();

  cv::Mat& getColorMask();
  cv::Mat& getIndexMask();

  int rows = 640;
  int cols = 1280;
  cv::Mat original_image;

protected:
  bool processOutput(const samplesCommon::BufferManager &buffers) override;
  bool processOutputColored(const samplesCommon::BufferManager &buffers);
  // the fast version does not work correctly right now
  bool processOutputFast(const samplesCommon::BufferManager &buffers);

  cv::Mat _colored_mask;
  cv::Mat _index_mask;
  bool _color_mask_ready = false;
  bool _index_mask_ready = false;

  std::map<int, vector<uint8_t> > _colors = {
      {0, {0, 0, 0}},
      {1, {0, 177, 247}},
      {2, {94, 30, 104}},
      {3, {191, 119, 56}},
      {4, {40, 140, 40}},
      // {5, {146, 243, 146}},
      {5, {10, 250, 30}},
      {6, {250, 0, 55}},
      {7, {178, 20, 50}},
      {8, {0, 30, 130}},
      {9, {0, 255, 127}},
      {10, {243, 15, 190}}
  };
  float _alpha;
};

#endif //TRTWRAPPER_PROJ_TRT_SEGMENTATION_H
