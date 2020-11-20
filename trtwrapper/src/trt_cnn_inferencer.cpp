#include "trt_cnn_inferencer.h"

TRTCNNInferencer::TRTCNNInferencer() {
  m_builder = samplesCommon::InferObject(nvinfer1::createInferBuilder(gLogger));
  m_builder_config = samplesCommon::InferObject(m_builder->createBuilderConfig());

  // WARN: is that legal?
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  m_network_definition =
      samplesCommon::InferObject(m_builder->createNetworkV2(explicitBatch));

  m_onnx_parser = samplesCommon::InferObject(
      nvonnxparser::createParser(*m_network_definition, gLogger));
  m_uff_parser = samplesCommon::InferObject(nvuffparser::createUffParser());

  m_runtime = samplesCommon::InferObject(createInferRuntime(gLogger));

  initLibNvInferPlugins(&gLogger, "");
}

TRTCNNInferencer::~TRTCNNInferencer() {}

void TRTCNNInferencer::configureGPUMemory() {
  size_t free_bits;
  size_t total_bits;

  cudaError_t err = cudaMemGetInfo(&free_bits, &total_bits);
  if (err == cudaSuccess) {

    size_t free_bytes = free_bits / 1024;
    size_t total_bytes = total_bits / 1024;
    size_t bytes_fraction = total_bytes * m_gpu_memory_fraction;

    if (bytes_fraction > free_bytes) {
      std::cerr << "Warning, desired GPU memoryy fraction is lower than free "
                   "memory available. Will be set free fraction"
                << std::endl;
      m_builder_config->setMaxWorkspaceSize(free_bytes);
    } else {
      m_builder_config->setMaxWorkspaceSize(bytes_fraction);
    }

  } else {
    // Just guess
    m_builder_config->setMaxWorkspaceSize(2_GiB);
  }
}

bool TRTCNNInferencer::loadFromONNX(const std::string &filename) {

  m_is_loaded = false;

  if (filename.rfind(".onnx") == std::string::npos) {
    m_last_error = "File is not ONNX model";
    return false;
  }

  if (!FileExists(filename)) {
    m_last_error = "File not exists";
    return false;
  }

  clearModel();

  m_builder->setMaxBatchSize(m_batch_size);

  configureBitMode();
  configureGPUMemory();

  bool success = m_onnx_parser->parseFromFile(
      filename.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
  //, *m_network_definition, DataType::kFLOAT);
  if (!success) {
    m_last_error = "Can't parse model file " +
                  std::string(m_onnx_parser->getError(0)->desc());
    return false;
  }

  auto prof = m_builder->createOptimizationProfile();

  prof->setDimensions(
      getInputNodeName().c_str(), OptProfileSelector::kMIN,
      nvinfer1::Dims4{(int)m_batch_size, getInputHeight(), getInputWidth(), 3});
  prof->setDimensions(
      getInputNodeName().c_str(), OptProfileSelector::kMAX,
      nvinfer1::Dims4{(int)m_batch_size, getInputHeight(), getInputWidth(), 3});
  prof->setDimensions(
      getInputNodeName().c_str(), OptProfileSelector::kOPT,
      nvinfer1::Dims4{(int)m_batch_size, getInputHeight(), getInputWidth(), 3});

  m_builder_config->addOptimizationProfile(prof);

  m_cuda_engine = samplesCommon::InferObject(
      m_builder->buildEngineWithConfig(*m_network_definition, *m_builder_config));

  if (!m_cuda_engine) {
    m_last_error = "Can't build cuda engine";
    return false;
  }

  m_context = samplesCommon::InferObject(m_cuda_engine->createExecutionContext());

  if (!m_context) {
    m_last_error = "Can't create context";
    return false;
  }

  m_context->setBindingDimensions(
      0,
      nvinfer1::Dims4{(int)m_batch_size, getInputHeight(), getInputWidth(), 3});

  // Create RAII buffer manager object
  m_buffers = std::shared_ptr<samplesCommon::BufferManager>(
      new samplesCommon::BufferManager(m_cuda_engine, (int)m_batch_size));

  m_model_filename = filename;

  m_is_loaded = true;
  return true;
}

bool TRTCNNInferencer::loadFromCudaEngine(const string &filename) {
  m_is_loaded = false;

  if (filename.rfind(".ce") == std::string::npos &&
      filename.rfind(".bin") == std::string::npos) {
    m_last_error = "File is not Cuda Engine binary model";
    return false;
  }

  if (!FileExists(filename)) {
    m_last_error = "File not exists";
    return false;
  }

  clearModel();

  m_builder->setMaxBatchSize(m_batch_size);

//  configureBitMode();
//  configureGPUMemory();

  std::ifstream input(filename, std::ios::binary);
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

  m_cuda_engine = samplesCommon::InferObject(
      m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));

  if (!m_cuda_engine) {
    m_last_error = "Can't build cuda engine";

    return false;
  }

  m_context = samplesCommon::InferObject(m_cuda_engine->createExecutionContext());

  if (!m_context) {
    m_last_error = "Can't create context";
    return false;
  }

  // Create RAII buffer manager object
  m_buffers = std::shared_ptr<samplesCommon::BufferManager>(
      new samplesCommon::BufferManager(m_cuda_engine, m_batch_size));

  m_model_filename = filename;

  m_is_loaded = true;
  return true;
}

void TRTCNNInferencer::configureBitMode() {

  if (!m_builder) {
    return;
  }
  if (!m_builder_config) {
    return;
  }

  m_builder_config->clearFlag(nvinfer1::BuilderFlag::kINT8);
  m_builder_config->clearFlag(nvinfer1::BuilderFlag::kFP16);

  if (m_bit_mode == BIT_MODE::AUTO) {
    if (m_builder->platformHasFastInt8()) {
      m_bit_mode = BIT_MODE::INT8;
    } else if (m_builder->platformHasFastFp16()) {
      m_bit_mode = BIT_MODE::FLOAT16;
    } else {
      m_bit_mode = BIT_MODE::FLOAT32;
    }
    return configureBitMode();
  }

  else if (m_bit_mode == BIT_MODE::FLOAT16) {
    m_builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if (m_bit_mode == BIT_MODE::INT8) {
    m_builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
  } else if (m_bit_mode == BIT_MODE::FLOAT32) {
    // nothing, flags were skipped
  }
}

bool TRTCNNInferencer::loadFromUff(const string &filename) {
  m_is_loaded = false;

  if (filename.rfind(".uff") == std::string::npos) {
    m_last_error = "File is not UFF model";
    return false;
  }

  if (!FileExists(filename)) {
    m_last_error = "File not exists";
    return false;
  }

  clearModel();

  m_uff_parser->registerInput(m_input_node_name.c_str(), m_input_shape,
                             nvuffparser::UffInputOrder::kNHWC);

  for (size_t i = 0; i < m_output_node_names.size(); i++)
    m_uff_parser->registerOutput(m_output_node_names[i].c_str());

  m_builder->setMaxBatchSize((int)m_batch_size);
  configureGPUMemory();
  configureBitMode();

  bool success = m_uff_parser->parse(filename.c_str(), *m_network_definition,
                                    DataType::kFLOAT);
  if (!success) {
    m_last_error = "Can't parse model file";
    return false;
  }

  m_cuda_engine = samplesCommon::InferObject(
      m_builder->buildEngineWithConfig(*m_network_definition, *m_builder_config));

  if (!m_cuda_engine) {
    m_last_error = "Can't build cuda engine";
    return false;
  }

  m_context = samplesCommon::InferObject(m_cuda_engine->createExecutionContext());

  if (!m_context) {
    m_last_error = "Can't create context";
    return false;
  }

  // Create RAII buffer manager object
  m_buffers = std::shared_ptr<samplesCommon::BufferManager>(
      new samplesCommon::BufferManager(m_cuda_engine, m_batch_size));

  m_model_filename = filename;

  m_is_loaded = true;
  return true;
}

std::string TRTCNNInferencer::inference(const std::vector<cv::Mat> &imgs) {
  if (!m_cuda_engine) {
    m_last_error = "Not loaded model";
    return m_last_error;
  }

  if (!m_context) {
    m_last_error = "Can't create context";
    return m_last_error;
  }

#ifdef TRT_DEBUG
  auto t1 = std::chrono::high_resolution_clock::now();
#endif

  bool success =
      processInput(*m_buffers, imgs, {0.485, 0.456, 0.406,}, m_norm_type, m_bgr2rgb);
  m_buffers->copyInputToDevice();

  if (!success) {
    return m_last_error;
  }

  // NOTE: Execute V2?
  // "Current optimization profile is: 0." ??? Is it good? How to change?
  bool status =
      m_context->execute(m_batch_size, m_buffers->getDeviceBindings().data());

  if (!status) {
    m_last_error = "Can't execute context";
    return m_last_error;
  }

  // Memcpy from device output buffers to host output buffers
  m_buffers->copyOutputToHost();
  // WARN: and here we go to parse...

#ifdef TRT_DEBUG
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Inference took: " << duration << " microseconds" << std::endl;
#endif

  return "OK";
}

int TRTCNNInferencer::getInputHeight() const { return m_input_shape.d[0]; }

void TRTCNNInferencer::setInputHeight(const int &height) {
  m_input_shape.d[0] = height;
}

int TRTCNNInferencer::getInputWidth() const { return m_input_shape.d[1]; }

void TRTCNNInferencer::setInputWidth(const int &width) {
  m_input_shape.d[1] = width;
}

int TRTCNNInferencer::getInputDepth() const { return m_input_shape.d[2]; }

void TRTCNNInferencer::setInputDepth(const int &depth) {
  m_input_shape.d[2] = depth;
}

std::string TRTCNNInferencer::getLastError() const { return m_last_error; }

std::string TRTCNNInferencer::getModelFilename() const {
  return m_model_filename;
}

TRTCNNInferencer::BIT_MODE TRTCNNInferencer::getBitMode() const {
  return m_bit_mode;
}

void TRTCNNInferencer::setBitMode(const BIT_MODE &bit_mode) {
  m_bit_mode = bit_mode;
}

bool TRTCNNInferencer::processInput(const samplesCommon::BufferManager &buffers,
                                    const std::vector<cv::Mat> &imgs,
                                    const std::vector<float> &mean,
                                    NormalizeType normalize, bool rgb) {

  const int inputC = getInputDepth();
  const int inputH = getInputHeight();
  const int inputW = getInputWidth();

  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(m_input_node_name));
  // NOTE: Carefully, size is in bytes!
  const int size = buffers.size(m_input_node_name) / sizeof(float);

  if (!hostDataBuffer) {
    m_last_error = "Can not get input tensor by name: " + m_input_node_name;
    return false;
  }

  // Host memory for input buffer
  const int volImg = inputC * inputH * inputW;
  const int volChl = inputH * inputW;
  cv::Mat input_img;

  for (int i = 0; i < m_batch_size; ++i) {

    // Nearest is faster, but results are different
    cv::resize(imgs[i], input_img, cv::Size(inputW, inputH), 0, 0);

    // positions - height - width
    input_img.forEach<cv::Vec3b>([&](cv::Vec3b &pixel,
                                     const int position[]) -> void {
      for (short c = 0; c < inputC; ++c) {
        float val(pixel[c]);
        if (normalize == NormalizeType::SEGMENTATION) {
          val /= 255.0;
          val -= mean[2-c];
          val /= m_deviation[2-c];
        } else if (normalize == NormalizeType::DETECTION) {
          val = (2.0f / 255.0f) * val - 1.0f;
        } else if (normalize == NormalizeType::CLASSIFICATION_SLIM) {
          // WARN: IDK why, this shouldnt happen, but work only with this
          // preprocessing
          val = float(val);
          val /= 255.0;
          // val -= 0.5;
          // val *= 2.0;
        }

        int pos = 0;
        if (rgb) {
          pos = i * volImg + (2 - c) * volChl + position[0] * inputW + position[1];
        } else {
          pos = i * volImg + c * volChl + position[0] * inputW + position[1];
        }

        hostDataBuffer[pos] = val;
      }
    });
  }
#ifdef IMAGE_DUMP_DEBUG
  std::vector<float> check_(hostDataBuffer, hostDataBuffer + volImg);
  std::ofstream outFile("image_vec.txt");
  for (const auto &e : check_) outFile << e << "\n";
#endif

  return true;
}

void TRTCNNInferencer::clearModel() {
  m_builder->reset();
  m_builder_config->reset();
}

bool TRTCNNInferencer::getBGR2RGBConvertionEnabled() const { return m_bgr2rgb; }

void TRTCNNInferencer::setBGR2RGBConvertionEnabled(bool bgr2rgb) {
  m_bgr2rgb = bgr2rgb;
}

NormalizeType TRTCNNInferencer::getNormalizationType() const {
  return m_norm_type;
}

void TRTCNNInferencer::setNormalizationType(const NormalizeType &norm_type) {
  m_norm_type = norm_type;
}
