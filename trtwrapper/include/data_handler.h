#ifndef TRT_INFERENCE_DATA_HANDLER_H
#define TRT_INFERENCE_DATA_HANDLER_H

#include "interfaces.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include <fstream>
#include <map>
#include <opencv2/core/types.hpp>
#include <vector>

class DataHandling : public IDataBase {
public:
  DataHandling() = default;
  virtual ~DataHandling() = default;

  bool loadColors() override;

  bool loadConfig() override;

  bool loadDetectionLabels() override;

  bool setConfigPath(std::string path) override;

  bool setConfig(const ConfigData &config) override;

  cv::Size getConfigInputSize() override;

  std::string getConfigInputNode() override;

  std::vector<std::string> getConfigOutputNodes() override;

  std::string getConfigEnginePath() override;

  std::string getConfigColorsPath() override;

  std::vector<std::array<int, 3>> getColors() override;

  std::vector<std::string> getDetectionLabels() override;

  std::vector<float> getConfigCategoriesThresholds() override;

  bool setConfigInputSize(const cv::Size &size) override;

  bool setConfigInputNode(const std::string &input_node) override;

  bool
  setConfigOutputNodes(const std::vector<std::string> &output_node) override;

  bool setConfigEnginePath(const std::string &engine_path) override;

  bool setConfigColorsPath(const std::string &colors_path) override;

  bool setConfigCategoriesThresholds(const std::vector<float> &categories_thresholds) override;

protected:
  std::string m_config_path = "config.json";
  std::vector<std::array<int, 3>> m_colors;
  std::vector<std::string> m_detection_labels;
  ConfigData m_config;
  std::fstream m_config_datafile;

  bool openConfig();
  static std::string tryParseJsonMember(rapidjson::Document &doc,
                                        const std::string &name,
                                        const std::string &default_val = "");
};

#endif // TRT_INFERENCE_DATA_HANDLER_H
