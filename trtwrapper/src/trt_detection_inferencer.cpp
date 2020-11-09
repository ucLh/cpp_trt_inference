#include "trt_detection_inferencer.h"

TRTDetectionInferencer::TRTDetectionInferencer() {
  _norm_type = NormalizeType::CLASSIFICATION_SLIM;
  _bgr2rgb = true;
}

string TRTDetectionInferencer::inference(const std::vector<cv::Mat> &imgs) {

#ifdef TRT_DEBUG
  _frames.clear();
  _bb_frames.clear();

  for (auto fr : imgs) {
    cv::Mat t;
    fr.copyTo(t);
    _frames.push_back(t);
  }
#endif

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

#ifdef TRT_DEBUG
std::vector<cv::Mat>
TRTDetectionInferencer::getFramesWithBoundingBoxes(float tresh) {
  if (tresh == 0.0f) {
    tresh = _thresh;
  }

  if (_bb_frames.size()) {
    return _bb_frames;
  }

  for (short i = 0, total = _frames.size(); i < total; ++i) {
    cv::Mat frame;
    _frames[i].copyTo(frame);

    for (short j = 0; j < _boxes[i].size(); ++j) {

      if (_scores[i][j] < tresh)
        continue;

      const int x = _boxes[i][j].tl().x * (float)frame.cols;
      const int y = _boxes[i][j].tl().y * (float)frame.rows;
      const int x1 = _boxes[i][j].br().x * (float)frame.cols;
      const int y1 = _boxes[i][j].br().y * (float)frame.rows;
      cv::Scalar color = cv::Scalar(250, 250, 250);

      if (_label_colors.size() > _classes[i][j]) {
        const auto &cl = _label_colors[_classes[i][j]];
        color = cv::Scalar(cl[0], cl[1], cl[2]);
      }

      std::string name = "Object";
      if (_label_names.size() > _classes[i][j]) {
        name = _label_names[_classes[i][j]];
      }

      cv::rectangle(frame, cv::Point(x, y), cv::Point(x1, y1), color, 4);
      cv::putText(frame, name, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX,
                  2.0, color, 2);
    }

    _bb_frames.push_back(frame);
  }

  return _bb_frames;
}
#endif

bool TRTDetectionInferencer::processOutput(
    const samplesCommon::BufferManager &buffers) {
  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(_output_node_names[0]));
  // NOTE: buffers.size give bytes, not lenght, be carefull
  const size_t size =
      (buffers.size(_output_node_names[0]) / sizeof(float)) / _batch_size;

  if (!hostDataBuffer) {
    _last_error = "Can not get output tensor by name " + _output_node_names[0];
    return false;
  }

  /// parse it?

  if (size % _layout_size) {
    _last_error = "Number of outputs not correspond with layout size";
    return false;
  }

  _boxes.clear();
  _classes.clear();
  _scores.clear();

  _boxes.resize(_batch_size);
  _classes.resize(_batch_size);
  _scores.resize(_batch_size);

  const size_t number_of_detections = size / _layout_size;
  for (size_t example_num = 0; example_num < _batch_size; ++example_num) {

    _boxes[example_num].resize(number_of_detections);
    _scores[example_num].resize(number_of_detections);
    _classes[example_num].resize(number_of_detections);

    for (size_t index = 0; index < number_of_detections; ++index) {
      const size_t prefix =
          (example_num * number_of_detections) + index * _layout_size;

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

      _boxes[example_num][index] =
          cv::Rect2f(xmin, ymin, xmax - xmin, ymax - ymin);
      _scores[example_num][index] = score;
      _classes[example_num][index] = cl_index;
    }
  }

  return true;
}

std::vector<std::vector<int>> TRTDetectionInferencer::getClasses() const {
  return _classes;
}

std::vector<std::vector<float>> TRTDetectionInferencer::getScores() const {
  return _scores;
}

std::vector<std::vector<cv::Rect2f>> TRTDetectionInferencer::getBoxes() const {
  return _boxes;
}

float TRTDetectionInferencer::getThresh() const { return _thresh; }

void TRTDetectionInferencer::setThresh(float thresh) { _thresh = thresh; }
