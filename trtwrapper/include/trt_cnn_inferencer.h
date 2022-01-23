#ifndef TRT_CNN_INFERENCER_H
#define TRT_CNN_INFERENCER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvUtils.h"
#include "logger.h"

#include "trt_buffers.h"
#include "trt_common.h"
#include <fstream>
#include <iterator>
#include <ostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define TRT_DEBUG

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

enum NormalizeType {
  DETECTION,
  CLASSIFICATION_SLIM,
  SEGMENTATION,
  DETECTION_YOLOV4,
  DETECTION_EFFDET
};

///
/// \brief The TRTCNNInferencer class defines interface for TRT inferencers:
/// Detection, Classification, Segmentation, etc It's verrritual class do not
/// use it.
///
class TRTCNNInferencer {
public:
  TRTCNNInferencer();
  virtual ~TRTCNNInferencer();

  // No reason to support copy constructor
  TRTCNNInferencer(const TRTCNNInferencer &) = delete;
  TRTCNNInferencer(TRTCNNInferencer &&that) = default;

protected:
  ///
  /// \brief loadFromCudaEngine the only method that works for now. Can be
  /// loaded only on the same GPU or very similar Otherwise program can crash.
  /// Export now  works only with TRT on Python:
  /// https://github.com/AastaNV/TRT_object_detection
  /// \param filename
  /// \return true if all was ok. If not check getLastError method
  ///
  virtual bool loadFromCudaEngine(const std::string &filename);

  ///
  /// \brief inference
  /// \param imgs
  /// \return status string, if all will be "OK" or empty
  ///  If not check getLastError method
  virtual std::string inference(const std::vector<cv::Mat> &imgs);

  int getInputHeight() const;

  int getInputWidth() const;

  int getInputDepth() const;

  std::string getLastError() const;

  virtual bool processInput(const samplesCommon::BufferManager &buffers,
                            const std::vector<cv::Mat> &imgs,
                            const std::vector<float> &mean = {0, 0, 0},
                            NormalizeType normalize = NormalizeType::DETECTION,
                            bool rgb = true);

  virtual bool processOutput(const samplesCommon::BufferManager &buffers) = 0;

  virtual void clearModel();

  std::string m_last_error = "";

  std::shared_ptr<nvinfer1::IBuilder> m_builder = nullptr;

  nvinfer1::Dims3 m_input_shape = {608, 608, 3};

  std::shared_ptr<nvinfer1::IBuilderConfig> m_builder_config = nullptr;
  std::shared_ptr<nvinfer1::ICudaEngine> m_cuda_engine{nullptr};
  std::shared_ptr<IRuntime> m_runtime = nullptr;
  std::shared_ptr<IExecutionContext> m_context = nullptr;
  std::shared_ptr<samplesCommon::BufferManager> m_buffers;

  size_t m_batch_size = 1;

  std::vector<std::vector<int>> m_label_colors = {{0, 0, 0}};

  std::string m_input_node_name = "input";
  std::vector<std::string> m_output_node_names = {
      "nms_num_detections", "nms_boxes", "nms_scores", "nms_classes"};

  NormalizeType m_norm_type = NormalizeType::DETECTION;
  std::vector<float> m_deviation = {0.229, 0.224, 0.225};
  bool m_bgr2rgb = true; // otherwise RGB
};

#endif // TRT_CNN_INFERENCER_H
