#include "data_handler.h"
#include "csv/csv.h"
#include <sstream>
#include <string>
#include <utility>

bool DataHandling::openConfig() {
  m_config_datafile.open(m_config_path, std::ios::in | std::ios::app);
  return true;
}

std::string DataHandling::tryParseJsonMember(rapidjson::Document &doc,
                                             const std::string &name,
                                             const std::string &default_val) {
  if (doc.HasMember(name.c_str())) {
    rapidjson::Value &value = doc[name.c_str()];
    return value.GetString();
  } else {
    return default_val;
  }
}
template <class T>
std::vector<float> tryParseJsonArray(rapidjson::Document &doc,
                                          const std::string &name,
                                          const std::vector<T> &default_val = {}) {
  if (doc.HasMember(name.c_str())) {
    rapidjson::Value &value = doc[name.c_str()];
    auto generic_array = value.GetArray();
    std::vector<T> result_array;
    for (rapidjson::SizeType i = 0; i < generic_array.Size(); i++) {
      result_array.emplace_back(generic_array[i].Get<T>());
    }
    return result_array;
  } else {
    return default_val;
  }
}

bool DataHandling::loadConfig() {

  using namespace rapidjson;
  Document doc;
  std::string line;
  std::stringstream json_doc_buffer;

  openConfig();

  if (m_config_datafile.is_open()) {
    while (std::getline(m_config_datafile, line)) {
      json_doc_buffer << line << "\n";
    }

    doc.Parse(json_doc_buffer.str().c_str());
    if (doc.IsObject()) {
      rapidjson::Value &input_size = doc["input_size"];
      rapidjson::Value &input_node = doc["input_node"];
      rapidjson::Value &output_node = doc["output_nodes"];
      rapidjson::Value &engine_path = doc["engine_path"];

      m_config.input_node = input_node.GetString();
      for (const auto &node : output_node.GetArray()) {
        auto node_str = node.GetString();
        m_config.output_nodes.emplace_back(node_str);
      }
      m_config.engine_path = engine_path.GetString();
      m_config.input_size.height = input_size.GetArray()[0].GetInt();
      m_config.input_size.width = input_size.GetArray()[1].GetInt();
      m_config.colors_path = tryParseJsonMember(doc, "colors_path");
      return true;
    } else {
      return false;
    }

  } else {
    return false;
  }
}

bool DataHandling::loadColors() {
  io::CSVReader<4> in(m_config.colors_path);
  in.read_header(io::ignore_extra_column, "name", "r", "g", "b");
  std::string name;
  int r, g, b;
  while (in.read_row(name, r, g, b)) {
    std::array<int, 3> color = {r, g, b};
    m_colors.emplace_back(color);
  }

  return true;
}

bool DataHandling::loadDetectionLabels() {
  io::CSVReader<1> in(m_config.detection_labels_path);
  in.read_header(io::ignore_extra_column, "class");
  std::string label;
  while (in.read_row(label)) {
    m_detection_labels.emplace_back(label);
  }

  return true;
}

bool DataHandling::setConfigPath(std::string path) {
  if (path.empty()) {
    std::cerr << "Config path is empty!" << std::endl;
    return false;
  }
  m_config_path = std::move(path);
  return true;
}

bool DataHandling::setConfig(const IDataBase::ConfigData &config) {
  m_config = config;
  return true;
}

cv::Size DataHandling::getConfigInputSize() { return m_config.input_size; }

std::string DataHandling::getConfigInputNode() { return m_config.input_node; }

std::vector<std::string> DataHandling::getConfigOutputNodes() {
  return m_config.output_nodes;
}

std::string DataHandling::getConfigEnginePath() { return m_config.engine_path; }

std::string DataHandling::getConfigColorsPath() { return m_config.colors_path; }

std::vector<std::array<int, 3>> DataHandling::getColors() { return m_colors; }

std::vector<std::string> DataHandling::getDetectionLabels() {
  return m_detection_labels;
}

std::vector<float> DataHandling::getConfigCategoriesThresholds() {
  return m_config.categories_thresholds;
}

bool DataHandling::getConfigShowObjectClass() {
  return m_config.show_object_class;
}

bool DataHandling::setConfigInputSize(const cv::Size &size) {
  m_config.input_size = size;
  return true;
}

bool DataHandling::setConfigInputNode(const std::string &input_node) {
  m_config.input_node = input_node;
  return true;
}

bool DataHandling::setConfigOutputNodes(
    const std::vector<std::string> &output_nodes) {
  m_config.output_nodes = output_nodes;
  return true;
}

bool DataHandling::setConfigEnginePath(const std::string &embed_pb_path) {
  m_config.engine_path = embed_pb_path;
  return true;
}

bool DataHandling::setConfigColorsPath(const std::string &colors_path) {
  m_config.colors_path = colors_path;
  return true;
}

bool DataHandling::setConfigCategoriesThresholds(const std::vector<float> &categories_thresholds) {
  m_config.categories_thresholds = categories_thresholds;
  return true;
}

bool DataHandling::setConfigShowObjectClass(bool show_object_class) {
  m_config.show_object_class = show_object_class;
  return true;
}
