#include "trt_detection_inferencer.h"

TRTDetectionInferencer::TRTDetectionInferencer() {
  m_norm_type = NormalizeType::CLASSIFICATION_SLIM;
  m_bgr2rgb = true;
}

string TRTDetectionInferencer::inference(const std::vector<cv::Mat> &imgs) {

#ifdef TRT_DEBUG
  m_frames.clear();
  m_bb_frames.clear();

  for (auto fr : imgs) {
    cv::Mat t;
    fr.copyTo(t);
    m_frames.push_back(t);
  }
#endif

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

#ifdef TRT_DEBUG
std::vector<cv::Mat>
TRTDetectionInferencer::getFramesWithBoundingBoxes(float tresh) {
  if (tresh == 0.0f) {
    tresh = m_thresh;
  }

  if (m_bb_frames.size()) {
    return m_bb_frames;
  }

  for (short i = 0, total = m_frames.size(); i < total; ++i) {
    cv::Mat frame;
    m_frames[i].copyTo(frame);

    for (short j = 0; j < m_boxes[i].size(); ++j) {

      if (m_scores[i][j] < tresh)
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
      if (m_label_names.size() > m_classes[i][j]) {
        name = m_label_names[m_classes[i][j]];
      }

      cv::rectangle(frame, cv::Point(x, y), cv::Point(x1, y1), color, 4);
      cv::putText(frame, name, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX,
                  2.0, color, 2);
    }

    m_bb_frames.push_back(frame);
  }

  return m_bb_frames;
}
#endif

bool TRTDetectionInferencer::processOutput(
    const samplesCommon::BufferManager &buffers) {
  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(m_output_node_names[0]));
  // NOTE: buffers.size give bytes, not lenght, be carefull
  const size_t size =
      (buffers.size(m_output_node_names[0]) / sizeof(float)) / m_batch_size;

  if (!hostDataBuffer) {
    m_last_error = "Can not get output tensor by name " + m_output_node_names[0];
    return false;
  }

  /// parse it?

  if (size % m_layout_size) {
    m_last_error = "Number of outputs not correspond with layout size";
    return false;
  }

  m_boxes.clear();
  m_classes.clear();
  m_scores.clear();

  m_boxes.resize(m_batch_size);
  m_classes.resize(m_batch_size);
  m_scores.resize(m_batch_size);

  const size_t number_of_detections = size / m_layout_size;
  for (size_t example_num = 0; example_num < m_batch_size; ++example_num) {

    m_boxes[example_num].resize(number_of_detections);
    m_scores[example_num].resize(number_of_detections);
    m_classes[example_num].resize(number_of_detections);

    for (size_t index = 0; index < number_of_detections; ++index) {
      const size_t prefix =
          (example_num * number_of_detections) + index * m_layout_size;

      //            std::cout << hostDataBuffer[prefix +0] << " " <<
      //            hostDataBuffer[prefix + 1] << " " << hostDataBuffer[prefix
      //            +2] << " "
      //                                                      <<
      //                                                      hostDataBuffer[prefix
      //                                                      +3] << " "<<
      //                                                      hostDataBuffer[prefix
      //                                                      +4] << " "
      //                                                      <<
      //                                                      hostDataBuffer[prefix
      //                                                      +5] << " " <<
      //                                                      hostDataBuffer[prefix
      //                                                      +6] << std::endl;

      const int cl_index = static_cast<int>(hostDataBuffer[prefix + 1]);
      const float score = hostDataBuffer[prefix + 2];

      const float xmin = hostDataBuffer[prefix + 3];
      const float ymin = hostDataBuffer[prefix + 4];
      const float xmax = hostDataBuffer[prefix + 5];
      const float ymax = hostDataBuffer[prefix + 6];

      m_boxes[example_num][index] =
          cv::Rect2f(xmin, ymin, xmax - xmin, ymax - ymin);
      m_scores[example_num][index] = score;
      m_classes[example_num][index] = cl_index;
    }
  }

  return true;
}

std::vector<std::vector<int>> TRTDetectionInferencer::getClasses() const {
  return m_classes;
}

std::vector<std::vector<float>> TRTDetectionInferencer::getScores() const {
  return m_scores;
}

std::vector<std::vector<cv::Rect2f>> TRTDetectionInferencer::getBoxes() const {
  return m_boxes;
}

float TRTDetectionInferencer::getThresh() const { return m_thresh; }

void TRTDetectionInferencer::setThresh(float thresh) { m_thresh = thresh; }
