#include "data_handler.h"
#include "csv/csv.h"
#include <sstream>
#include <utility>
#include <string>

bool DataHandling::open_config() {
  config_datafile_.open(config_path, std::ios::in | std::ios::app);
  return true;
}

std::string
DataHandling::try_parse_json_member(rapidjson::Document &doc,
                                    const std::string &name,
                                    const std::string &default_val) {
  if (doc.HasMember(name.c_str())) {
    rapidjson::Value &value = doc[name.c_str()];
    return value.GetString();
  } else {
    return default_val;
  }
}

bool DataHandling::load_config() {

  using namespace rapidjson;
  Document doc;
  std::string line;
  std::stringstream json_doc_buffer;

  open_config();

  if (config_datafile_.is_open()) {
    while (std::getline(config_datafile_, line)) {
      json_doc_buffer << line << "\n";
    }

    doc.Parse(json_doc_buffer.str().c_str());
    if (doc.IsObject()) {
      rapidjson::Value &input_size = doc["input_size"];
      rapidjson::Value &input_node = doc["input_node"];
      rapidjson::Value &output_node = doc["output_node"];
      rapidjson::Value &engine_path = doc["engine_path"];

      config.input_node = input_node.GetString();
      config.output_node = output_node.GetString();
      config.engine_path = engine_path.GetString();
      config.input_size.height = input_size.GetArray()[0].GetInt();
      config.input_size.width = input_size.GetArray()[1].GetInt();
      config.colors_path = try_parse_json_member(doc, "colors_path");
      return true;
    } else {
      return false;
    }

  } else {
    return false;
  }
}

bool DataHandling::load_colors() {
  io::CSVReader<4> in(config.colors_path);
  in.read_header(io::ignore_extra_column, "name", "r", "g", "b");
  std::string name;
  int r, g, b;
  while (in.read_row(name, r, g, b)) {
    std::array<int, 3> color = {r, g, b};
    colors.emplace_back(color);
  }

  return true;
}

bool DataHandling::set_config_path(std::string path) {
  if (path.empty()) {
    std::cerr << "Config path is empty!" << std::endl;
    return false;
  }
  config_path = std::move(path);
  return true;
}

cv::Size DataHandling::get_config_input_size() { return config.input_size; }

std::string DataHandling::get_config_input_node() { return config.input_node; }

std::string DataHandling::get_config_output_node() {
  return config.output_node;
}

std::string DataHandling::get_config_engine_path() {
  return config.engine_path;
}

std::vector<std::array<int, 3>> DataHandling::get_colors() { return colors; }

bool DataHandling::set_config_input_size(const cv::Size &size) {
  config.input_size = size;
  return true;
}

bool DataHandling::set_config_input_node(const std::string &input_node) {
  config.input_node = input_node;
  return true;
}

bool DataHandling::set_config_output_node(const std::string &output_node) {
  config.output_node = output_node;
  return true;
}

bool DataHandling::set_config_engine_path(const std::string &embed_pb_path) {
  config.engine_path = embed_pb_path;
  return true;
}

bool DataHandling::set_config_colors_path(const std::string &colors_path) {
  config.colors_path = colors_path;
  return true;
}
