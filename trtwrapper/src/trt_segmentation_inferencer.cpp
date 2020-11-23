#include "trt_segmentation_inferencer.h"
#include "data_handler.h"
#include <map>
#include <memory>
#include <vector>

TRTSegmentationInferencer::TRTSegmentationInferencer() {
  m_data_handler = make_unique<DataHandling>();
  m_norm_type = NormalizeType::SEGMENTATION;
  m_bgr2rgb = true;
  m_colored_mask = cv::Mat(m_rows, m_cols, CV_8UC3);
  m_index_mask = cv::Mat(m_rows, m_cols, CV_8UC1);
}

string TRTSegmentationInferencer::inference(const std::vector<cv::Mat> &imgs) {
  if (!m_ready_for_inference) {
    return "You need to call prepareForInference first.";
  }

  std::string status = TRTCNNInferencer::inference(imgs);

  if (status.size() >= 2) {
    return status;
  }
}

bool TRTSegmentationInferencer::prepareForInference(
    const std::string &config_path) {
  m_data_handler->setConfigPath(config_path);
  m_data_handler->loadConfig();

  setInputNodeName(m_data_handler->getConfigInputNode());
  setOutputNodeName({m_data_handler->getConfigOutputNode()});

  m_rows = m_data_handler->getConfigInputSize().height;
  m_cols = m_data_handler->getConfigInputSize().width;
  m_input_shape = {m_rows, m_cols, 3};

  m_data_handler->loadColors();
  m_colors = m_data_handler->getColors();
  m_num_classes_actual = m_colors.size();

  TRTCNNInferencer::loadFromCudaEngine(m_data_handler->getConfigEnginePath());
  m_ready_for_inference = true;
  return true;
}

string TRTSegmentationInferencer::makeIndexMask() {
  bool ok = processOutputArgmaxed(*m_buffers);

  if (!ok) {
    return m_last_error;
  }
  m_index_mask_ready = true;
  return "OK";
}

string TRTSegmentationInferencer::makeColorMask(float alpha,
                                                const cv::Mat &original_image) {
  bool ok = processOutputColoredArgmaxed(*m_buffers, alpha, original_image);

  if (!ok) {
    return m_last_error;
  }
  m_color_mask_ready = true;
  return "OK";
}

void *TRTSegmentationInferencer::getHostDataBuffer() {
  auto output_node_name = getOutputNodeName()[0];
  return m_buffers->getHostBuffer(output_node_name);
}

std::size_t TRTSegmentationInferencer::getHostDataBufferBytesNum() {
  auto output_node_name = getOutputNodeName()[0];
  return m_buffers->size(output_node_name);
}

bool TRTSegmentationInferencer::processOutputColored(
    const samplesCommon::BufferManager &buffers, float alpha,
    const cv::Mat &original_image) {
  auto output_node_name = getOutputNodeName()[0];
  auto *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(output_node_name));
  const size_t num_of_elements = getHostDataBufferSize<float>();
  assert(num_of_elements % (m_rows * m_cols) == 0);

  if (!hostDataBuffer) {
    m_last_error = "Can not get output tensor by name " + output_node_name;
    return false;
  }

  int size = m_rows * m_cols;
  size_t num_channels = num_of_elements / (size);
  cv::Mat img;
  cv::resize(original_image, img, cv::Size(m_cols, m_rows), 0, 0);
  img = img.reshape(0, 1);
  //  cv::Vec3b pixel;
  //  uint8_t *indexes_ptr = indexes.data;

  std::vector<float> point(num_channels);
  for (int i = 0; i != size; ++i) {
    for (auto j = 0; j != num_channels; ++j) {
      auto offset = size * j + i;
      point[j] = hostDataBuffer[offset];
    }
    // find argmax
    int maxElementIndex =
        std::max_element(point.begin(), point.end()) - point.begin();

    cv::Vec3b &pixel = m_colored_mask.at<cv::Vec3b>(cv::Point(i, 0));
    cv::Vec3b img_pixel = img.at<cv::Vec3b>(cv::Point(i, 0));
    pixel[0] =
        (1 - alpha) * m_colors[maxElementIndex][2] + alpha * img_pixel[2];
    pixel[1] =
        (1 - alpha) * m_colors[maxElementIndex][1] + alpha * img_pixel[1];
    pixel[2] =
        (1 - alpha) * m_colors[maxElementIndex][0] + alpha * img_pixel[0];
    std::fill(point.begin(), point.end(), -1);
  }

  m_colored_mask = m_colored_mask.reshape(0, m_rows);

  return true;
}

bool TRTSegmentationInferencer::processOutputColoredFast(
    const samplesCommon::BufferManager &buffers, float alpha,
    const cv::Mat &original_image) {
  auto output_node_name = getOutputNodeName()[0];
  auto *hostDataBuffer =
      static_cast<half_float::half *>(buffers.getHostBuffer(output_node_name));
  const size_t num_of_elements = getHostDataBufferSize<half_float::half>();
  assert(num_of_elements % (m_rows * m_cols) == 0);

  if (!hostDataBuffer) {
    m_last_error = "Can not get output tensor by name " + output_node_name;
    return false;
  }
  //  int img_size = rows * cols;
  //  size_t num_channels = num_of_elements / (img_size);

  // Needs to be a multiple of 8. If the actual number
  // is lower, the remaining channels will be filled with zeros
  // See https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-format-desc__fig4
  const int num_classes = 16;

  cv::Mat net_output(m_rows, m_cols, CV_16FC(num_classes), hostDataBuffer);
  uint8_t *mask_ptr = m_colored_mask.data;
  cv::Mat img;
  cv::resize(original_image, img, cv::Size(m_cols, m_rows), 0, 0);
  uint8_t *img_ptr = img.data;
  typedef cv::Vec<cv::float16_t, num_classes> Vecnb;
  net_output.forEach<Vecnb>([&](Vecnb &pixel, const int position[]) -> void {
    std::vector<float> p{pixel.val, pixel.val + m_num_classes_actual};
    int maxElementIndex = std::max_element(p.begin(), p.end()) - p.begin();
    int hw_pos = position[0] * m_cols + position[1];
    mask_ptr[3 * hw_pos + 0] = (1 - alpha) * m_colors[maxElementIndex][2] +
                               alpha * img_ptr[3 * hw_pos + 0];
    mask_ptr[3 * hw_pos + 1] = (1 - alpha) * m_colors[maxElementIndex][1] +
                               alpha * img_ptr[3 * hw_pos + 1];
    mask_ptr[3 * hw_pos + 2] = (1 - alpha) * m_colors[maxElementIndex][0] +
                               alpha * img_ptr[3 * hw_pos + 2];
  });

  return true;
}

bool TRTSegmentationInferencer::processOutput(
    const samplesCommon::BufferManager &buffers) {
  auto output_node_name = getOutputNodeName()[0];
  auto *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(output_node_name));
  const size_t num_of_elements = getHostDataBufferSize<float>();
  assert(num_of_elements % (m_rows * m_cols) == 0);

  if (!hostDataBuffer) {
    m_last_error = "Can not get output tensor by name " + output_node_name;
    return false;
  }

  int size = m_rows * m_cols;
  size_t num_channels = num_of_elements / (size);
  uint8_t *indexes_ptr = m_index_mask.data;
  int maxElementIndex = -1;

  std::vector<float> point(num_channels);
  for (int i = 0; i != size; ++i) {
    for (auto j = 0; j != num_channels; ++j) {
      auto offset = size * j + i;
      point[j] = hostDataBuffer[offset];
    }
    // find argmax
    maxElementIndex =
        std::max_element(point.begin(), point.end()) - point.begin();
    indexes_ptr[0, i] = maxElementIndex;
  }

  m_index_mask = m_index_mask.reshape(0, m_rows);

  return true;
}

bool TRTSegmentationInferencer::processOutputFast(
    const samplesCommon::BufferManager &buffers) {
  auto output_node_name = getOutputNodeName()[0];
  auto *hostDataBuffer =
      static_cast<half_float::half *>(buffers.getHostBuffer(output_node_name));
  const size_t num_of_elements = getHostDataBufferSize<half_float::half>();
  assert(num_of_elements % (m_rows * m_cols) == 0);

  if (!hostDataBuffer) {
    m_last_error = "Can not get output tensor by name " + output_node_name;
    return false;
  }
  //  int img_size = rows * cols;
  //  size_t num_channels = num_of_elements / (img_size);

  // Needs to be a multiple of 8. If the actual number
  // is lower, the remaining channels will be filled with zeros
  const int num_classes = 16;

  cv::Mat net_output(m_rows, m_cols, CV_16FC(num_classes), hostDataBuffer);
  uint8_t *indexes_ptr = m_index_mask.data;
  typedef cv::Vec<cv::float16_t, num_classes> Vecnb;
  net_output.forEach<Vecnb>([&](Vecnb &pixel, const int position[]) -> void {
    std::vector<float> p{pixel.val, pixel.val + m_num_classes_actual};
    int maxElementIndex = std::max_element(p.begin(), p.end()) - p.begin();
    indexes_ptr[0, position[0] * m_cols + position[1]] = maxElementIndex;
  });

  return true;
}

bool TRTSegmentationInferencer::processOutputArgmaxed(
    const samplesCommon::BufferManager &buffers) {
  auto output_node_name = getOutputNodeName()[0];
  auto *hostDataBuffer =
      static_cast<int *>(buffers.getHostBuffer(output_node_name));
  const size_t num_of_elements = getHostDataBufferSize<int>();
  assert(num_of_elements % (m_rows * m_cols) == 0);

  if (!hostDataBuffer) {
    m_last_error = "Can not get output tensor by name " + output_node_name;
    return false;
  }

  std::vector<int> data_vec{hostDataBuffer, hostDataBuffer + num_of_elements};
  m_index_mask = cv::Mat(data_vec).reshape(1, m_rows);
  return true;
}

bool TRTSegmentationInferencer::processOutputColoredArgmaxed(
    const samplesCommon::BufferManager &buffers, float alpha,
    const cv::Mat &original_image) {
  auto output_node_name = getOutputNodeName()[0];
  auto *hostDataBuffer =
      static_cast<int *>(buffers.getHostBuffer(output_node_name));
  const size_t num_of_elements = getHostDataBufferSize<int>();
  assert(num_of_elements % (m_rows * m_cols) == 0);

  if (!hostDataBuffer) {
    m_last_error = "Can not get output tensor by name " + output_node_name;
    return false;
  }

  std::vector<int> data_vec{hostDataBuffer, hostDataBuffer + num_of_elements};
  m_index_mask = cv::Mat(data_vec).reshape(1, m_rows);
  // Call to the processOutputArgmaxed does not give the expected result right
  // now,
  // so for now we are repeating the code.

  uint8_t *mask_ptr = m_colored_mask.data;
  cv::Mat img;
  cv::resize(original_image, img, cv::Size(m_cols, m_rows), 0, 0);
  uint8_t *img_ptr = img.data;

  m_index_mask.forEach<cv::Vec<int, 1>>(
      [&](cv::Vec<int, 1> pixel, const int position[]) -> void {
        int hw_pos = position[0] * m_cols + position[1];
        mask_ptr[3 * hw_pos + 0] = (1 - alpha) * m_colors[pixel.val[0]][2] +
                                   alpha * img_ptr[3 * hw_pos + 0];
        mask_ptr[3 * hw_pos + 1] = (1 - alpha) * m_colors[pixel.val[0]][1] +
                                   alpha * img_ptr[3 * hw_pos + 1];
        mask_ptr[3 * hw_pos + 2] = (1 - alpha) * m_colors[pixel.val[0]][0] +
                                   alpha * img_ptr[3 * hw_pos + 2];
      });

  return true;
}

cv::Mat &TRTSegmentationInferencer::getColorMask() {
  if (m_color_mask_ready) {
    return m_colored_mask;
  } else {
    std::cerr << "You have to call makeColorMask() first!"
              << "\n";
    exit(1);
  }
}

cv::Mat &TRTSegmentationInferencer::getIndexMask() {
  if (m_index_mask_ready) {
    return m_index_mask;
  } else {
    std::cerr << "You have to call makeIndexMask() first!"
              << "\n";
    exit(1);
  }
}

std::string TRTSegmentationInferencer::getLastError() {
  return TRTCNNInferencer::getLastError();
}
