#include "trt_segmentation_inferencer.h"
#include <map>
#include <vector>

TRTSegmentationInferencer::TRTSegmentationInferencer() {
  _input_node_name = "input_0";
  _output_node_names = {"output_0"};

  _norm_type = NormalizeType::CLASSIFICATION_SLIM;
  _bgr2rgb = true;
}

string TRTSegmentationInferencer::inference(const std::vector<cv::Mat> &imgs) {

  std::string status = TRTCNNInferencer::inference(imgs);

  if (status.size() > 2) {
    return status;
  }

  bool ok = processOutputIndexed(*_buffers);

  if (!ok) {
    return _last_error;
  }

  return "OK";
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
  float alpha = 0.4;
  size_t num_channels = num_of_elements / (size);
  cv::Mat colored_mask(1, size, CV_8UC3);
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

    cv::Vec3b &pixel = colored_mask.at<cv::Vec3b>(cv::Point(i, 0));
    cv::Vec3b img_pixel = img.at<cv::Vec3b>(cv::Point(i, 0));
    pixel[0] = (1 - alpha) * _colors[maxElementIndex][2] + alpha * img_pixel[2];
    pixel[1] = (1 - alpha) * _colors[maxElementIndex][1] + alpha * img_pixel[1];
    pixel[2] = (1 - alpha) * _colors[maxElementIndex][0] + alpha * img_pixel[0];
//    colored_mask.at<cv::Vec3b>(cv::Point(i, 0)) = pixel;
//    indexes.data[i] = maxElementIndex; alpha * img_original + (1 - alpha) * color_map
    std::fill(point.begin(), point.end(), -1);
  }

  _colored_mask = colored_mask.reshape(0, rows);
  
  //  for (auto i = 0; i != num_channels; ++i) {
  //    int start = i * size;
  //    int finish = (i + 1) * size - 1;
  //    std::vector<float> temp(hostDataBuffer + start, hostDataBuffer +
  //    finish); cv::Mat probs_mat(rows, cols, CV_32FC1); memcpy(probs_mat.data,
  //    temp.data(), temp.size() * sizeof(float));
  //    _probs.emplace_back(probs_mat);
  //  }
  return true;
}

bool TRTSegmentationInferencer::processOutputIndexed(
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
  float alpha = 0.4;
  size_t num_channels = num_of_elements / (size);
  cv::Mat index_mask(1, size, CV_8UC1);
  uint8_t *indexes_ptr = index_mask.data;
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

  _colored_mask = index_mask.reshape(0, rows);

  return true;
}

bool TRTSegmentationInferencer::processOutputIndexedFast(
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
  const int num_classes = 11;
  size_t num_channels = num_of_elements / (size);
  cv::Mat net_output(rows, cols,  CV_32FC(num_classes), hostDataBuffer);
  cv::Mat index_mask(rows, cols, CV_8UC1);
  uint8_t *indexes_ptr = index_mask.data;
  typedef cv::Vec<float, num_classes> Vecnb;
  net_output.forEach<Vecnb>([&](Vecnb &pixel,
                                   const int position[]) -> void {
    std::vector<float> p{pixel.val, pixel.val + num_classes};
    int maxElementIndex = std::max_element(p.begin(), p.end()) - p.begin();
    indexes_ptr[position[0] * cols + position[1]] = maxElementIndex;
  });

  _colored_mask = index_mask;

  return true;
}


std::vector<float> TRTSegmentationInferencer::getScores() const {
  return _scores;
}

std::vector<int> TRTSegmentationInferencer::getClasses() const {
  return _classes;
}

float TRTSegmentationInferencer::getThresh() const { return _thresh; }

void TRTSegmentationInferencer::setThresh(float thresh) { _thresh = thresh; }

cv::Mat TRTSegmentationInferencer::getColoredMask() { return _colored_mask; }
