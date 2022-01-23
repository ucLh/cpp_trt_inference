#ifndef TRT_DETECTION_INFERENCER_H
#define TRT_DETECTION_INFERENCER_H

#if CV_VERSION_MAJOR >= 4
  #include <opencv4/opencv2/opencv.hpp>
#else
  #include <opencv2/opencv.hpp>
#endif
#include <string>
#include <vector>

#include "interfaces.h"
#include "trt_cnn_inferencer.h"

///
/// \brief The TRTDetectionInferencer class inferencing detection networks
///
class TRTDetectionInferencer : public virtual TRTCNNInferencer,
                               public virtual IDetectionInferenceHandler {
public:
  TRTDetectionInferencer();

  TRTDetectionInferencer(TRTDetectionInferencer &&that);
  ~TRTDetectionInferencer() override = default;

  bool prepareForInference(const IDataBase::ConfigData &config) override;
  std::string inference(const std::vector<cv::Mat> &imgs) override;
  std::string getLastError() const override;

  //#ifdef TRT_DEBUG
  ///
  /// \brief getFramesWithBoundingBoxes. Debug visualization high-level method
  /// \param tresh
  /// \return Image with drawn bounding boxes
  ///
  std::vector<cv::Mat>
  getFramesWithBoundingBoxes(const std::vector<cv::Mat> &imgs) override;
  //#endif

  std::vector<std::vector<cv::Rect2f>> getBoxes() const override;
  std::vector<std::vector<float>> getScores() const override;
  std::vector<std::vector<int>> getClasses() const override;

  float getThresh() const override;
  void setThresh(float thresh) override;

protected:
  bool processConfig();
  bool processOutput(const samplesCommon::BufferManager &buffers) override;
  int postprocessOutput(int cl_index, float score,
                        const std::vector<float> &tresholds = {0.2, 0.1, 0.1,
                                                               0.2, 0.3});
  virtual int remapClassIndex(int cl_index) = 0;
  static bool filterScore(int cl_index, float score,
                          const vector<float> &tresholds);
  virtual cv::Rect2f processBox(float xmin, float ymin, float xmax, float ymax,
                          int index) = 0;
  template <class T>
  bool checkBufferExtraction(T const &buffer, int output_node_name_index) {
    if (!buffer) {
      m_last_error = "Can not get output tensor by name " +
                     m_output_node_names[output_node_name_index];
      return false;
    }
    return true;
  }

  bool m_ready_for_inference = false;
  bool m_inference_completed = false;
  bool m_new_inference_happend;
  bool m_show_object_class;

  std::unique_ptr<IDataBase> m_data_handler;
  int m_rows;
  int m_cols;
  std::vector<float> m_current_original_rows; // Len is equal to batch size
  std::vector<float> m_current_original_cols; // Len is equal to batch size
  std::vector<std::string> m_detection_labels;

  //  size_t m_layout_size = 7;
  float m_thresh = 0.1;
  std::vector<float> m_categories_thresholds;

  std::vector<std::vector<cv::Rect2f>> m_boxes;
  std::vector<std::vector<float>> m_scores;
  std::vector<std::vector<int>> m_classes;

#ifdef TRT_DEBUG
  std::vector<cv::Mat> m_bb_frames;
  std::vector<cv::Mat> m_frames;
#endif
};

#endif // TRT_DETECTION_INFERENCER_H
