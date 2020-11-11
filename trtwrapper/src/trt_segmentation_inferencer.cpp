#include "trt_segmentation_inferencer.h"
#include "data_handler.h"
#include <map>
#include <memory>
#include <vector>

TRTSegmentationInferencer::TRTSegmentationInferencer() {
  data_handler_ = make_unique<DataHandling>();
  _norm_type = NormalizeType::SEGMENTATION;
  _bgr2rgb = true;
  colored_mask_ = cv::Mat(rows_, cols_, CV_8UC3);
  index_mask_ = cv::Mat(rows_, cols_, CV_8UC1);
}

string TRTSegmentationInferencer::inference(const std::vector<cv::Mat> &imgs) {
  if (!ready_for_inference_) {
    return "You need to call prepareForInference first.";
  }

  std::string status = TRTCNNInferencer::inference(imgs);

  if (status.size() >= 2) {
    return status;
  }
}

bool TRTSegmentationInferencer::prepareForInference(
    const std::string &config_path) {
  data_handler_->set_config_path(config_path);
  data_handler_->load_config();

  input_node_name_ = data_handler_->get_config_input_node();
  output_node_names_ = {data_handler_->get_config_output_node()};

  rows_ = data_handler_->get_config_input_size().height;
  cols_ = data_handler_->get_config_input_size().width;

  data_handler_->load_colors();
  colors_ = data_handler_->get_colors();
  num_classes_actual_ = colors_.size();

  TRTCNNInferencer::loadFromCudaEngine(data_handler_->get_config_engine_path());
  ready_for_inference_ = true;
  return true;
}

string TRTSegmentationInferencer::makeIndexMask() {
  bool ok = processOutputFast(*_buffers);

  if (!ok) {
    return _last_error;
  }
  index_mask_ready_ = true;
  return "OK";
}

string TRTSegmentationInferencer::makeColorMask(float alpha,
                                                const cv::Mat &original_image) {
  bool ok = processOutputColoredFast(*_buffers, alpha, original_image);

  if (!ok) {
    return _last_error;
  }
  color_mask_ready_ = true;
  return "OK";
}

bool TRTSegmentationInferencer::processOutputColored(
    const samplesCommon::BufferManager &buffers, float alpha,
    const cv::Mat &original_image) {
  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(output_node_names_[0]));
  // NOTE: buffers.size give bytes, not length, be careful
  const size_t num_of_elements =
      (buffers.size(output_node_names_[0]) / sizeof(float)) / _batch_size;

  if (!hostDataBuffer) {
    _last_error = "Can not get output tensor by name " + output_node_names_[0];
    return false;
  }

  int size = rows_ * cols_;
  size_t num_channels = num_of_elements / (size);
  cv::Mat img;
  cv::resize(original_image, img, cv::Size(cols_, rows_), 0, 0);
  img = img.reshape(0, 1);
  //  cv::Vec3b pixel;
  //  uint8_t *indexes_ptr = indexes.data;

  std::vector<float> point(num_channels);
  for (int i = 0; i != size; ++i) {
    for (auto j = 0; j != num_channels; ++j) {
      auto offset = size * j + i;
      point[j] = hostDataBuffer[offset];
    }
    // find argmax
    int maxElementIndex =
        std::max_element(point.begin(), point.end()) - point.begin();

    cv::Vec3b &pixel = colored_mask_.at<cv::Vec3b>(cv::Point(i, 0));
    cv::Vec3b img_pixel = img.at<cv::Vec3b>(cv::Point(i, 0));
    pixel[0] = (1 - alpha) * colors_[maxElementIndex][2] + alpha * img_pixel[2];
    pixel[1] = (1 - alpha) * colors_[maxElementIndex][1] + alpha * img_pixel[1];
    pixel[2] = (1 - alpha) * colors_[maxElementIndex][0] + alpha * img_pixel[0];
    std::fill(point.begin(), point.end(), -1);
  }

  colored_mask_ = colored_mask_.reshape(0, rows_);

  return true;
}

bool TRTSegmentationInferencer::processOutputColoredFast(
    const samplesCommon::BufferManager &buffers, float alpha,
    const cv::Mat &original_image) {
  auto *hostDataBuffer = static_cast<half_float::half *>(
      buffers.getHostBuffer(output_node_names_[0]));
  // NOTE: buffers.size give bytes, not length, be careful
  const size_t num_of_elements =
      (buffers.size(output_node_names_[0]) / sizeof(half_float::half)) /
      _batch_size;

  if (!hostDataBuffer) {
    _last_error = "Can not get output tensor by name " + output_node_names_[0];
    return false;
  }
  //  int img_size = rows * cols;
  //  size_t num_channels = num_of_elements / (img_size);

  // Needs to be a multiple of 8. If the actual number
  // is lower, the remaining channels will be filled with zeros
  const int num_classes = 16;

  cv::Mat net_output(rows_, cols_, CV_16FC(num_classes), hostDataBuffer);
  uint8_t *mask_ptr = colored_mask_.data;
  cv::Mat img;
  cv::resize(original_image, img, cv::Size(cols_, rows_), 0, 0);
  uint8_t *img_ptr = img.data;
  typedef cv::Vec<cv::float16_t, num_classes> Vecnb;
  net_output.forEach<Vecnb>([&](Vecnb &pixel, const int position[]) -> void {
    std::vector<float> p{pixel.val, pixel.val + 11};
    int maxElementIndex = std::max_element(p.begin(), p.end()) - p.begin();
    int hw_pos = position[0] * cols_ + position[1];
    mask_ptr[3 * hw_pos + 0] = (1 - alpha) * colors_[maxElementIndex][2] +
                               alpha * img_ptr[3 * hw_pos + 0];
    mask_ptr[3 * hw_pos + 1] = (1 - alpha) * colors_[maxElementIndex][1] +
                               alpha * img_ptr[3 * hw_pos + 1];
    mask_ptr[3 * hw_pos + 2] = (1 - alpha) * colors_[maxElementIndex][0] +
                               alpha * img_ptr[3 * hw_pos + 2];
  });

  return true;
}

bool TRTSegmentationInferencer::processOutput(
    const samplesCommon::BufferManager &buffers) {
  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(output_node_names_[0]));
  // NOTE: buffers.size give bytes, not length, be careful
  const size_t num_of_elements =
      (buffers.size(output_node_names_[0]) / sizeof(float)) / _batch_size;

  if (!hostDataBuffer) {
    _last_error = "Can not get output tensor by name " + output_node_names_[0];
    return false;
  }

  int size = rows_ * cols_;
  size_t num_channels = num_of_elements / (size);
  uint8_t *indexes_ptr = index_mask_.data;
  int maxElementIndex = -1;

  std::vector<float> point(num_channels);
  for (int i = 0; i != size; ++i) {
    for (auto j = 0; j != num_channels; ++j) {
      auto offset = size * j + i;
      point[j] = hostDataBuffer[offset];
    }
    // find argmax
    maxElementIndex =
        std::max_element(point.begin(), point.end()) - point.begin();
    indexes_ptr[0, i] = maxElementIndex;
  }

  index_mask_ = index_mask_.reshape(0, rows_);

  return true;
}

bool TRTSegmentationInferencer::processOutputFast(
    const samplesCommon::BufferManager &buffers) {
  auto *hostDataBuffer = static_cast<half_float::half *>(
      buffers.getHostBuffer(output_node_names_[0]));
  // NOTE: buffers.size give bytes, not length, be careful
  const size_t num_of_elements =
      (buffers.size(output_node_names_[0]) / sizeof(half_float::half)) /
      _batch_size;

  if (!hostDataBuffer) {
    _last_error = "Can not get output tensor by name " + output_node_names_[0];
    return false;
  }
  //  int img_size = rows * cols;
  //  size_t num_channels = num_of_elements / (img_size);

  // Needs to be a multiple of 8. If the actual number
  // is lower, the remaining channels will be filled with zeros
  const int num_classes = 16;

  cv::Mat net_output(rows_, cols_, CV_16FC(num_classes), hostDataBuffer);
  uint8_t *indexes_ptr = index_mask_.data;
  typedef cv::Vec<cv::float16_t, num_classes> Vecnb;
  net_output.forEach<Vecnb>([&](Vecnb &pixel, const int position[]) -> void {
    std::vector<float> p{pixel.val, pixel.val + 11};
    int maxElementIndex = std::max_element(p.begin(), p.end()) - p.begin();
    indexes_ptr[0, position[0] * cols_ + position[1]] = maxElementIndex;
  });

  return true;
}

cv::Mat &TRTSegmentationInferencer::getColorMask() {
  if (color_mask_ready_) {
    return colored_mask_;
  } else {
    std::cerr << "You have to call makeColorMask() first!"
              << "\n";
    exit(1);
  }
}

cv::Mat &TRTSegmentationInferencer::getIndexMask() {
  if (index_mask_ready_) {
    return index_mask_;
  } else {
    std::cerr << "You have to call makeIndexMask() first!"
              << "\n";
    exit(1);
  }
}

void TRTSegmentationInferencer::setMixingCoefficient(float alpha) {
  alpha_ = alpha;
}

std::string TRTSegmentationInferencer::getLastError() {
  return TRTCNNInferencer::getLastError();
}
