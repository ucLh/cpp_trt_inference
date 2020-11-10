#include "trt_segmentation_inferencer.h"
#include <map>
#include <vector>

TRTSegmentationInferencer::TRTSegmentationInferencer() {
  _input_node_name = "input_0";
  _output_node_names = {"output_0"};

  _norm_type = NormalizeType::SEGMENTATION;
  _bgr2rgb = true;
  _colored_mask = cv::Mat(1, rows * cols, CV_8UC3);
  _index_mask = cv::Mat(1, rows * cols, CV_8UC1);
}

string TRTSegmentationInferencer::inference(const std::vector<cv::Mat> &imgs) {
  std::string status = TRTCNNInferencer::inference(imgs);

  if (status.size() > 2) {
    return status;
  }
}

string TRTSegmentationInferencer::makeIndexMask() {
  bool ok = processOutput(*_buffers);

  if (!ok) {
    return _last_error;
  }
  _index_mask_ready = true;
  return "OK";
}

string TRTSegmentationInferencer::makeColorMask() {
  bool ok = processOutputColored(*_buffers);

  if (!ok) {
    return _last_error;
  }
  _color_mask_ready = true;
  return "OK";
}

bool TRTSegmentationInferencer::processOutputColored(
    const samplesCommon::BufferManager &buffers) {
  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(_output_node_names[0]));
  // NOTE: buffers.size give bytes, not length, be careful
  const size_t num_of_elements =
      (buffers.size(_output_node_names[0]) / sizeof(float)) / _batch_size;

  if (!hostDataBuffer) {
    _last_error = "Can not get output tensor by name " + _output_node_names[0];
    return false;
  }

  int size = rows * cols;
  size_t num_channels = num_of_elements / (size);
  cv::Mat img;
  cv::resize(original_image, img, cv::Size(cols, rows), 0, 0);
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

    cv::Vec3b &pixel = _colored_mask.at<cv::Vec3b>(cv::Point(i, 0));
    cv::Vec3b img_pixel = img.at<cv::Vec3b>(cv::Point(i, 0));
    pixel[0] = (1 - _alpha) * _colors[maxElementIndex][2] + _alpha * img_pixel[2];
    pixel[1] = (1 - _alpha) * _colors[maxElementIndex][1] + _alpha * img_pixel[1];
    pixel[2] = (1 - _alpha) * _colors[maxElementIndex][0] + _alpha * img_pixel[0];
    std::fill(point.begin(), point.end(), -1);
  }

  _colored_mask = _colored_mask.reshape(0, rows);

  return true;
}

bool TRTSegmentationInferencer::processOutput(
    const samplesCommon::BufferManager &buffers) {
  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(_output_node_names[0]));
  // NOTE: buffers.size give bytes, not length, be careful
  const size_t num_of_elements =
      (buffers.size(_output_node_names[0]) / sizeof(float)) / _batch_size;

  if (!hostDataBuffer) {
    _last_error = "Can not get output tensor by name " + _output_node_names[0];
    return false;
  }

  int size = rows * cols;
  size_t num_channels = num_of_elements / (size);
  uint8_t *indexes_ptr = _index_mask.data;
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

  _index_mask = _index_mask.reshape(0, rows);

  return true;
}

bool TRTSegmentationInferencer::processOutputFast(
    const samplesCommon::BufferManager &buffers) {
  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(_output_node_names[0]));
  // NOTE: buffers.size give bytes, not length, be careful
  const size_t num_of_elements =
      (buffers.size(_output_node_names[0]) / sizeof(float)) / _batch_size;

  if (!hostDataBuffer) {
    _last_error = "Can not get output tensor by name " + _output_node_names[0];
    return false;
  }

  const int num_classes = 11;
  size_t num_channels = num_of_elements / (rows * cols);
//  const int num_classes = (int) num_channels;

  cv::Mat net_output(rows, cols,  CV_32FC(num_classes), hostDataBuffer);
  cv::Mat index_mask(rows, cols, CV_8UC1);
  uint8_t *indexes_ptr = index_mask.data;
  typedef cv::Vec<float, num_classes> Vecnb;
  net_output.forEach<Vecnb>([&](Vecnb &pixel,
                                   const int position[]) -> void {
    std::vector<float> p{pixel.val, pixel.val + num_channels};
    std::vector<int> check(position, position + 2);
    int maxElementIndex = std::max_element(p.begin(), p.end()) - p.begin();
    indexes_ptr[position[0] * cols + position[1]] = maxElementIndex;
  });

  _colored_mask = index_mask;

  return true;
}

cv::Mat& TRTSegmentationInferencer::getColorMask() {
  if (_color_mask_ready) {
    return _colored_mask;
  }
  else {
    std::cerr << "You have to call makeColorMask() first!" << "\n";
    exit(1);
  }
}

cv::Mat& TRTSegmentationInferencer::getIndexMask() {
  if (_index_mask_ready) {
    return _index_mask;
  }
  else {
    std::cerr << "You have to call makeIndexMask() first!" << "\n";
    exit(1);
  }
}

void TRTSegmentationInferencer::setMixingCoefficient(float alpha) {
  _alpha = alpha;
}
