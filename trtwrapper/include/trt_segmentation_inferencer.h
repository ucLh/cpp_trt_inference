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
class TRTSegmentationInferencer : virtual public TRTCNNInferencer,
                                  virtual public ISegmentationInferenceHandler {
public:
  TRTSegmentationInferencer();

  TRTSegmentationInferencer(TRTSegmentationInferencer &&that);
  ~TRTSegmentationInferencer() = default;

  bool prepareForInference(const std::string &config_path) override;
  bool prepareForInference(const IDataBase::ConfigData &config) override;
  std::string inference(const std::vector<cv::Mat> &imgs) override;
  std::string makeIndexMask(int pixel_sky_border) override;
  std::string makeColorMask(float alpha, const cv::Mat &original_image,
                            int pixel_sky_border) override;

  cv::Mat getColorMask() override;
  cv::Mat getIndexMask() override;

  std::size_t getHostDataBufferBytesNum() override;
  void *getHostDataBuffer() override;

  std::string getLastError() const override;

protected:
  bool processConfig();

  // NOTE: buffers.size give bytes, not length, be careful
  template <class T> size_t getHostDataBufferSize() {
    auto output_node_name = getOutputNodeName()[0];
    return (m_buffers->size(output_node_name) / sizeof(T)) / m_batch_size;
  }
  bool processOutput(const samplesCommon::BufferManager &buffers) override;
  bool processOutputColored(const samplesCommon::BufferManager &buffers,
                            float alpha, const cv::Mat &original_image);
  bool processOutputFast(const samplesCommon::BufferManager &buffers);
  bool processOutputColoredFast(const samplesCommon::BufferManager &buffers,
                                float alpha, const cv::Mat &original_image);
  bool processOutputArgmaxed(const samplesCommon::BufferManager &buffers,
                             int pixel_sky_border);
  bool processOutputColoredArgmaxed(const samplesCommon::BufferManager &buffers,
                                    float alpha, const cv::Mat &original_image,
                                    int pixel_sky_border);

  std::unique_ptr<IDataBase> m_data_handler;
  int m_rows = 640;
  int m_cols = 1280;
  int m_original_rows = 0;
  int m_original_cols = 0;
  int m_num_classes_actual;

  cv::Mat m_colored_mask;
  cv::Mat m_index_mask;
  bool m_color_mask_ready = false;
  bool m_index_mask_ready = false;
  bool m_ready_for_inference = false;
  bool m_inference_completed = false;

  std::vector<std::array<int, 3>> m_colors;
};

#endif // TRTWRAPPER_PROJ_TRT_SEGMENTATION_H
