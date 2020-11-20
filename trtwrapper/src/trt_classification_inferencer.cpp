#include "trt_classification_inferencer.h"

TRTClassificationInferencer::TRTClassificationInferencer() {
  m_input_node_name = "input:0";
  m_output_node_names = {"MobilenetV1/Predictions/Reshape_1:0"};

  m_norm_type = NormalizeType::CLASSIFICATION_SLIM;
  m_bgr2rgb = true;
}

string
TRTClassificationInferencer::inference(const std::vector<cv::Mat> &imgs) {

  std::string status = TRTCNNInferencer::inference(imgs);

  if (status.size() > 2) {
    return status;
  }

  bool ok = processOutput(*m_buffers);

  if (!ok) {
    return m_last_error;
  }

  return "OK";
}

bool TRTClassificationInferencer::processOutput(
    const samplesCommon::BufferManager &buffers) {
  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(m_output_node_names[0]));
  // NOTE: buffers.size give bytes, not length, be careful
  const size_t num_of_classes =
      (buffers.size(m_output_node_names[0]) / sizeof(float)) / m_batch_size;

  if (!hostDataBuffer) {
    m_last_error = "Can not get output tensor by name " + m_output_node_names[0];
    return false;
  }

  m_classes.clear();
  m_scores.clear();

  for (size_t example_num = 0; example_num < m_batch_size; ++example_num) {

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

    m_classes.push_back(best_class);
    m_scores.push_back(best_score);
  }

  return true;
}

std::vector<float> TRTClassificationInferencer::getScores() const {
  return m_scores;
}

std::vector<int> TRTClassificationInferencer::getClasses() const {
  return m_classes;
}

float TRTClassificationInferencer::getThresh() const { return m_thresh; }

void TRTClassificationInferencer::setThresh(float thresh) { m_thresh = thresh; }
