#include "trt_detection_inferencer.h"
#include "data_handler.h"
#include <memory>

TRTDetectionInferencer::TRTDetectionInferencer() {
  m_data_handler = std::make_unique<DataHandling>();
  m_norm_type = NormalizeType::DETECTION_YOLOV4;
  m_bgr2rgb = true;
}

bool TRTDetectionInferencer::prepareForInference(
    const DataHandling::ConfigData &config) {
  m_data_handler->setConfig(config);
  processConfig();

  if (m_ready_for_inference) {
    std::cerr << "Warning! You are preparing for inference multiple times"
              << "\n";
  }
  m_ready_for_inference = true;
  return true;
}

bool TRTDetectionInferencer::processConfig() {
  setInputNodeName(m_data_handler->getConfigInputNode());
  setOutputNodeNames({m_data_handler->getConfigOutputNodes()});

  m_rows = m_data_handler->getConfigInputSize().height;
  m_cols = m_data_handler->getConfigInputSize().width;
  m_input_shape = {m_rows, m_cols, 3};

  m_data_handler->loadDetectionLabels();
  m_detection_labels = m_data_handler->getDetectionLabels();
  m_categories_thresholds = m_data_handler->getConfigCategoriesThresholds();
  assert(m_detection_labels.size() == m_categories_thresholds.size() &&
         "The number of thresholds must be equal to the number of labels for "
         "detection");

  TRTCNNInferencer::loadFromCudaEngine(m_data_handler->getConfigEnginePath());
}

string TRTDetectionInferencer::inference(const std::vector<cv::Mat> &imgs) {
  if (!m_ready_for_inference) {
    m_last_error = "You need to call prepareForInference first!";
    return m_last_error;
  }
  std::string status = TRTCNNInferencer::inference(imgs);

  bool ok = processOutput(*m_buffers);
  m_inference_completed = true;
  m_new_inference_happend = true;

  if (!ok) {
    return m_last_error;
  }

  return status;
}

//#ifdef TRT_DEBUG
std::vector<cv::Mat> TRTDetectionInferencer::getFramesWithBoundingBoxes(
    const std::vector<cv::Mat> &imgs) {
  if (!m_inference_completed) {
    std::cerr << "You need to call inference() first!";
    exit(1);
  }

  if (m_new_inference_happend) {
    m_frames.clear();
    m_bb_frames.clear();

    for (const auto &fr : imgs) {
      cv::Mat t;
      fr.copyTo(t);
      m_frames.push_back(t);
    }
    m_new_inference_happend = false;
  }

  if (!m_bb_frames.empty()) {
    return m_bb_frames;
  }

  for (short i = 0, total = m_frames.size(); i < total; ++i) {
    cv::Mat frame;
    m_frames[i].copyTo(frame);

    for (short j = 0; j < m_boxes[i].size(); ++j) {

      if (m_scores[i][j] < m_thresh)
        continue;

      const int x = m_boxes[i][j].tl().x * (float)frame.cols;
      const int y = m_boxes[i][j].tl().y * (float)frame.rows;
      const int x1 = m_boxes[i][j].br().x * (float)frame.cols;
      const int y1 = m_boxes[i][j].br().y * (float)frame.rows;
      cv::Scalar color = cv::Scalar(250, 250, 250);

      if (m_label_colors.size() > m_classes[i][j]) {
        const auto &cl = m_label_colors[m_classes[i][j]];
        color = cv::Scalar(cl[0], cl[1], cl[2]);
      }

      std::string name = "Object";
      if (m_detection_labels.size() > m_classes[i][j]) {
        name = m_detection_labels[m_classes[i][j]];
      }

      cv::rectangle(frame, cv::Point(x, y), cv::Point(x1, y1), color, 4);
      cv::putText(frame, name, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX,
                  2.0, color, 2);
    }

    m_bb_frames.push_back(frame);
  }

  return m_bb_frames;
}
//#endif

bool TRTDetectionInferencer::processOutput(
    const samplesCommon::BufferManager &buffers) {
  auto *num_detections =
      static_cast<int *>(buffers.getHostBuffer(m_output_node_names[0]));
  auto *nms_boxes =
      static_cast<float *>(buffers.getHostBuffer(m_output_node_names[1]));
  auto *nms_scores =
      static_cast<float *>(buffers.getHostBuffer(m_output_node_names[2]));
  auto *nms_classes =
      static_cast<float *>(buffers.getHostBuffer(m_output_node_names[3]));

  // Check the extraction
  float *buffers_temp_array[] = {nms_boxes, nms_scores, nms_classes};
  if (!checkBufferExtraction<int *>(num_detections, 0)) {
    return false;
  }
  for (size_t i = 0; i < 3; ++i) {
    if (!checkBufferExtraction<float *>(buffers_temp_array[i], i)) {
      return false;
    }
  }

  // Parse buffers for drawing
  m_boxes.clear();
  m_classes.clear();
  m_scores.clear();

  m_boxes.resize(m_batch_size);
  m_classes.resize(m_batch_size);
  m_scores.resize(m_batch_size);

  const size_t number_of_detections = *num_detections;
  for (size_t example_num = 0; example_num < m_batch_size; ++example_num) {

    for (size_t index = 0; index < number_of_detections; ++index) {
      const int cl_index = nms_classes[index];
      const float score = nms_scores[index];
      const int final_cl_index = postprocessOutput(cl_index, score, m_categories_thresholds);

      const float xmin = nms_boxes[4 * index];
      const float ymin = nms_boxes[4 * index + 1];
      const float xmax = nms_boxes[4 * index + 2];
      const float ymax = nms_boxes[4 * index + 3];

      if (final_cl_index != -1) {
        m_boxes[example_num].emplace_back(
            cv::Rect2f(xmin, ymin, xmax - xmin, ymax - ymin));
        m_scores[example_num].emplace_back(score);
        m_classes[example_num].emplace_back(final_cl_index);
      }
    }
  }

  return true;
}

int TRTDetectionInferencer::remapClassIndex(int cl_index) {
  int final_cl_index = cl_index;
  if ((1 <= cl_index) && (cl_index < 9)) {
    // Vehicles
    final_cl_index = 1;
  } else if ((14 <= cl_index) && (cl_index < 24)) {
    // Animals
    final_cl_index = 2;
  } else if (cl_index == 32) {
    // Sport ball
    final_cl_index = 3;
  } else if (cl_index != 0) {
    // Other, not person
    final_cl_index = 4;
  }
  return final_cl_index;
}

bool TRTDetectionInferencer::filterScore(int cl_index, float score, const std::vector<float> &thresholds) {
  return score > thresholds[cl_index];
}

int TRTDetectionInferencer::postprocessOutput(int cl_index, float score,
                                              const std::vector<float> &thresholds) {
  /// Applies remapping and class-specific score filtering.
  /// returns: new class index after remapping if the filtering is passed,
  /// '-1' otherwise.
  int new_cl_index = remapClassIndex(cl_index);
  // No thresholds were given, don't apply filtering
  if (thresholds.empty()) {
    return new_cl_index;
  }
  bool keep = filterScore(new_cl_index, score, thresholds);
  if (keep) {
    return new_cl_index;
  } else {
    return -1;
  }
}

std::vector<std::vector<int>> TRTDetectionInferencer::getClasses() const {
  if (!m_inference_completed) {
    std::cerr << "You need to call inference() first!";
    exit(1);
  }
  return m_classes;
}

std::vector<std::vector<float>> TRTDetectionInferencer::getScores() const {
  if (!m_inference_completed) {
    std::cerr << "You need to call inference() first!";
    exit(1);
  }
  return m_scores;
}

std::vector<std::vector<cv::Rect2f>> TRTDetectionInferencer::getBoxes() const {
  if (!m_inference_completed) {
    std::cerr << "You need to call inference() first!";
    exit(1);
  }
  return m_boxes;
}

float TRTDetectionInferencer::getThresh() const { return m_thresh; }

void TRTDetectionInferencer::setThresh(float thresh) { m_thresh = thresh; }

std::string TRTDetectionInferencer::getLastError() const {
  return TRTCNNInferencer::getLastError();
}