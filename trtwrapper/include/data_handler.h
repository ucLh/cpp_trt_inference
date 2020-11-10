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

class DataHandling : public IDataBase{
public:
  DataHandling() = default;
  virtual ~DataHandling() = default;

  struct config_data {
    cv::Size input_size;
    std::string input_node;
    std::string output_node;
    std::string engine_path;
    std::string colors_path;
  };

  std::vector<std::array<int, 3> > colors;
  config_data config;
  std::string config_path = "config.json";

  bool load_colors() override;
  bool load_config() override ;

  bool set_config_path(std::string path) override;

  cv::Size get_config_input_size() override;

  std::string get_config_input_node() override;

  std::string get_config_output_node() override;

  std::string get_config_engine_path() override;

  std::vector<std::array<int, 3>> get_colors() override;

  bool set_config_input_size(const cv::Size &size) override;

  bool set_config_input_node(const std::string &input_node) override;

  bool set_config_output_node(const std::string &output_node) override;

  bool set_config_engine_path(const std::string &engine_path) override;

  bool set_config_colors_path(const std::string &colors_path) override;

protected:
  std::fstream config_datafile_;

  bool open_config();
  static std::string try_parse_json_member(rapidjson::Document &doc,
                                           const std::string &name,
                                           const std::string &default_val = "");
};

#endif // TRT_INFERENCE_DATA_HANDLER_H
