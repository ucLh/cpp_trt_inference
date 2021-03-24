#ifndef TRT_INFERENCE_INTERFACES_H
#define TRT_INFERENCE_INTERFACES_H

#include <opencv2/opencv.hpp>
#include <string>

class IDataBase {
public:
  struct ConfigData {
    cv::Size input_size;
    std::string input_node;
    std::vector<std::string> output_nodes;
    std::string engine_path;
    std::string colors_path;
    std::string detection_labels_path;
  };

  virtual bool setConfigPath(std::string path) = 0;

  virtual bool setConfig(const ConfigData &config) = 0;

  virtual bool loadConfig() = 0;

  virtual bool loadColors() = 0;

  virtual bool loadDetectionLabels() = 0;

  virtual cv::Size getConfigInputSize() = 0;

  virtual std::string getConfigInputNode() = 0;

  virtual std::vector<std::string> getConfigOutputNodes() = 0;

  virtual std::string getConfigEnginePath() = 0;

  virtual std::string getConfigColorsPath() = 0;

  virtual std::vector<std::array<int, 3>> getColors() = 0;

  virtual std::vector<std::string> getDetectionLabels() = 0;

  virtual bool setConfigInputSize(const cv::Size &size) = 0;

  virtual bool setConfigInputNode(const std::string &input_node) = 0;

  virtual bool
  setConfigOutputNodes(const std::vector<std::string> &output_nodes) = 0;

  virtual bool setConfigEnginePath(const std::string &engine_path) = 0;

  virtual bool setConfigColorsPath(const std::string &colors_path) = 0;
};

class IInferenceHandler {
public:
  virtual std::string getLastError() const = 0;
};

class ISegmentationInferenceHandler : public IInferenceHandler {
public:
  virtual bool prepareForInference(const std::string &config_path) = 0;

  virtual bool prepareForInference(const IDataBase::ConfigData &config) = 0;

  virtual std::string inference(const std::vector<cv::Mat> &imgs) = 0;

  virtual std::string makeIndexMask(int pixel_sky_border) = 0;

  virtual std::string makeColorMask(float alpha, const cv::Mat &original_image,
                                    int pixel_sky_border) = 0;

  virtual cv::Mat getIndexMask() = 0;

  virtual cv::Mat getColorMask() = 0;

  virtual void *getHostDataBuffer() = 0;

  virtual size_t getHostDataBufferBytesNum() = 0;
};

class IDetectionInferenceHandler : public IInferenceHandler {
public:
  virtual bool prepareForInference(const IDataBase::ConfigData &config) = 0;

  virtual std::string inference(const std::vector<cv::Mat> &imgs,
                                bool apply_postprocessing) = 0;

  virtual void setThresh(float thresh) = 0;

  virtual float getThresh() const = 0;

  virtual std::vector<cv::Mat>
  getFramesWithBoundingBoxes(const std::vector<cv::Mat> &imgs) = 0;

  virtual std::vector<std::vector<int>> getClasses() const = 0;

  virtual std::vector<std::vector<float>> getScores() const = 0;

  virtual std::vector<std::vector<cv::Rect2f>> getBoxes() const = 0;
};

#endif // TRT_INFERENCE_INTERFACES_H
