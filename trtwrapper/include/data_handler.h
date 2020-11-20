#ifndef TRT_INFERENCE_DATA_HANDLER_H
#define TRT_INFERENCE_DATA_HANDLER_H

#include <map>
#include <fstream>
#include <vector>
#include <opencv2/core/types.hpp>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "interfaces.h"

class DataHandling : public IDataBase {
public:
  DataHandling() = default;
  virtual ~DataHandling() = default;

  bool loadColors() override;

  bool loadConfig() override;

  bool setConfigPath(std::string path) override;

  cv::Size getConfigInputSize() override;

  std::string getConfigInputNode() override;

  std::string getConfigOutputNode() override;

  std::string getConfigEnginePath() override;

  std::vector<std::array<int, 3>> getColors() override;

  bool setConfigInputSize(const cv::Size &size) override;

  bool setConfigInputNode(const std::string &input_node) override;

  bool setConfigOutputNode(const std::string &output_node) override;

  bool setConfigEnginePath(const std::string &engine_path) override;

  bool setConfigColorsPath(const std::string &colors_path) override;

protected:
  struct configData {
    cv::Size input_size;
    std::string input_node;
    std::string output_node;
    std::string engine_path;
    std::string colors_path;
  };

  std::string m_config_path = "config.json";
  std::vector<std::array<int, 3>> m_colors;
  configData m_config;
  std::fstream m_config_datafile;

  bool openConfig();
  static std::string tryParseJsonMember(rapidjson::Document &doc,
                                        const std::string &name,
                                        const std::string &default_val = "");
};

#endif // TRT_INFERENCE_DATA_HANDLER_H
