#include "trt_segmentation.h"
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

  bool ok = processOutput(*_buffers, imgs);
  // return "Not OK";

  if (!ok) {
    return _last_error;
  }

  return "OK";
}

bool TRTSegmentationInferencer::processOutput(
    const samplesCommon::BufferManager &buffers, const std::vector<cv::Mat> &imgs) {
  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(_output_node_names[0]));
  // NOTE: buffers.size give bytes, not length, be careful
  const size_t num_of_elements =
      (buffers.size(_output_node_names[0]) / sizeof(float)) / _batch_size;

  if (!hostDataBuffer) {
    _last_error = "Can not get output tensor by name " + _output_node_names[0];
    return false;
  }

  std::map<int, vector<uint8_t> > colors = {
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

  std::vector<float> outputs(hostDataBuffer, hostDataBuffer + num_of_elements);
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
    pixel[0] = (1 - alpha) * colors[maxElementIndex][2] + alpha * img_pixel[2];
    pixel[1] = (1 - alpha) * colors[maxElementIndex][1] + alpha * img_pixel[1];
    pixel[2] = (1 - alpha) * colors[maxElementIndex][0] + alpha * img_pixel[0];
//    colored_mask.at<cv::Vec3b>(cv::Point(i, 0)) = pixel;
//    indexes.data[i] = maxElementIndex; alpha * img_original + (1 - alpha) * color_map
    std::fill(point.begin(), point.end(), -1);
  }
  std::cout << "H1" << std::endl;

  colored_mask = colored_mask.reshape(0, rows);
  
  //  for (auto i = 0; i != num_channels; ++i) {
  //    int start = i * size;
  //    int finish = (i + 1) * size - 1;
  //    std::vector<float> temp(hostDataBuffer + start, hostDataBuffer +
  //    finish); cv::Mat probs_mat(rows, cols, CV_32FC1); memcpy(probs_mat.data,
  //    temp.data(), temp.size() * sizeof(float));
  //    _probs.emplace_back(probs_mat);
  //  }
  cv::imwrite("1_trt_mix.jpg", colored_mask);
  return true;
}

bool TRTSegmentationInferencer::processOutput(
    const samplesCommon::BufferManager &buffers) {
  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(_output_node_names[0]));
  // NOTE: buffers.size give bytes, not length, be careful
  const size_t num_of_classes =
      (buffers.size(_output_node_names[0]) / sizeof(float)) / _batch_size;

  if (!hostDataBuffer) {
    _last_error = "Can not get output tensor by name " + _output_node_names[0];
    return false;
  }

  _classes.clear();
  _scores.clear();

  for (size_t example_num = 0; example_num < _batch_size; ++example_num) {

    int best_class = -1;
    float best_score = -1;
    for (size_t index = 0; index < num_of_classes; ++index) {
      const size_t prefix = (example_num * num_of_classes) + index;

      const float score = static_cast<float>(hostDataBuffer[prefix]);
      // std::cout << "Score: " << score << std::endl;
      if (score > best_score) {
        best_class = static_cast<int>(index);
        best_score = score;
      }
    }

    _classes.push_back(best_class);
    _scores.push_back(best_score);
  }

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
