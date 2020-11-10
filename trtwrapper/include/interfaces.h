#ifndef TRT_INFERENCE_INTERFACES_H
#define TRT_INFERENCE_INTERFACES_H

#include <opencv2/opencv.hpp>
#include <string>

class ISegmentationInferenceHandler {
public:
  virtual bool prepareForInference(const std::string &config_path) = 0;

  virtual std::string inference(const std::vector<cv::Mat> &imgs) = 0;

  virtual std::string makeIndexMask() = 0;

  virtual std::string makeColorMask(float alpha,
                                    const cv::Mat &original_image) = 0;

  virtual cv::Mat &getIndexMask() = 0;

  virtual cv::Mat &getColorMask() = 0;

  virtual bool loadFromCudaEngine(const std::string &filename) = 0;

  virtual std::string getLastError() = 0;
};

class IDataBase {
public:
  virtual bool set_config_path(std::string path) = 0;

  virtual bool load_config() = 0;

  virtual bool load_colors() = 0;

  virtual cv::Size get_config_input_size() = 0;

  virtual std::string get_config_input_node() = 0;

  virtual std::string get_config_output_node() = 0;

  virtual std::string get_config_engine_path() = 0;

  virtual std::vector<std::array<int, 3>> get_colors() = 0;

  virtual bool set_config_input_size(const cv::Size &size) = 0;

  virtual bool set_config_input_node(const std::string &input_node) = 0;

  virtual bool set_config_output_node(const std::string &output_node) = 0;

  virtual bool set_config_engine_path(const std::string &engine_path) = 0;

  virtual bool set_config_colors_path(const std::string &colors_path) = 0;
};

#endif // TRT_INFERENCE_INTERFACES_H
