#include "trt_classification_inferencer.h"

TRTClassificationInferencer::TRTClassificationInferencer() {
  input_node_name_ = "input:0";
  output_node_names_ = {"MobilenetV1/Predictions/Reshape_1:0"};

  _norm_type = NormalizeType::CLASSIFICATION_SLIM;
  _bgr2rgb = true;
}

string
TRTClassificationInferencer::inference(const std::vector<cv::Mat> &imgs) {

  std::string status = TRTCNNInferencer::inference(imgs);

  if (status.size() > 2) {
    return status;
  }

  bool ok = processOutput(*_buffers);

  if (!ok) {
    return _last_error;
  }

  return "OK";
}

bool TRTClassificationInferencer::processOutput(
    const samplesCommon::BufferManager &buffers) {
  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(output_node_names_[0]));
  // NOTE: buffers.size give bytes, not length, be careful
  const size_t num_of_classes =
      (buffers.size(output_node_names_[0]) / sizeof(float)) / _batch_size;

  if (!hostDataBuffer) {
    _last_error = "Can not get output tensor by name " + output_node_names_[0];
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

std::vector<float> TRTClassificationInferencer::getScores() const {
  return _scores;
}

std::vector<int> TRTClassificationInferencer::getClasses() const {
  return _classes;
}

float TRTClassificationInferencer::getThresh() const { return _thresh; }

void TRTClassificationInferencer::setThresh(float thresh) { _thresh = thresh; }
