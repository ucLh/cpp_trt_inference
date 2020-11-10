#ifndef TRTWRAPPER_PROJ_TRT_SEGMENTATION_H
#define TRTWRAPPER_PROJ_TRT_SEGMENTATION_H

#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "interfaces.h"
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

  bool prepareForInference(const std::string &config_path);
  std::string inference(const std::vector<cv::Mat> &imgs) override;
  std::string makeIndexMask();
  std::string makeColorMask(const cv::Mat &original_image);

  cv::Mat &getColorMask();
  cv::Mat &getIndexMask();

  void setMixingCoefficient(float alpha);

protected:
  bool processOutput(const samplesCommon::BufferManager &buffers) override;
  bool processOutputColored(const samplesCommon::BufferManager &buffers,
                            const cv::Mat &original_image);
  // the fast version does not work correctly right now
  bool processOutputFast(const samplesCommon::BufferManager &buffers);

  std::unique_ptr<IDataBase> data_handler_;
  int rows = 640;
  int cols = 1280;
  cv::Mat colored_mask_;
  cv::Mat index_mask_;
  bool color_mask_ready_ = false;
  bool index_mask_ready_ = false;
  bool ready_for_inference_ = false;

  std::vector<std::array<int, 3>> colors_;
  float alpha_;
};

#endif // TRTWRAPPER_PROJ_TRT_SEGMENTATION_H
