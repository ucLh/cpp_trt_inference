#include "trt_cnn_inferencer.h"

TRTCNNInferencer::TRTCNNInferencer() {
  _builder = samplesCommon::InferObject(nvinfer1::createInferBuilder(gLogger));
  _builder_config = samplesCommon::InferObject(_builder->createBuilderConfig());

  // WARN: is that legal?
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  _network_definition =
      samplesCommon::InferObject(_builder->createNetworkV2(explicitBatch));

  _onnx_parser = samplesCommon::InferObject(
      nvonnxparser::createParser(*_network_definition, gLogger));
  _uff_parser = samplesCommon::InferObject(nvuffparser::createUffParser());

  _runtime = samplesCommon::InferObject(createInferRuntime(gLogger));

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
    size_t bytes_fraction = total_bytes * _gpu_memory_fraction;

    if (bytes_fraction > free_bytes) {
      std::cerr << "Warning, desired GPU memoryy fraction is lower than free "
                   "memory available. Will be set free fraction"
                << std::endl;
      _builder_config->setMaxWorkspaceSize(free_bytes);
    } else {
      _builder_config->setMaxWorkspaceSize(bytes_fraction);
    }

  } else {
    // Just guess
    _builder_config->setMaxWorkspaceSize(2_GiB);
  }
}

bool TRTCNNInferencer::loadFromONNX(const std::string &filename) {

  _is_loaded = false;

  if (filename.rfind(".onnx") == std::string::npos) {
    _last_error = "File is not ONNX model";
    return false;
  }

  if (!FileExists(filename)) {
    _last_error = "File not exists";
    return false;
  }

  clearModel();

  _builder->setMaxBatchSize(_batch_size);

  configureBitMode();
  configureGPUMemory();

  bool success = _onnx_parser->parseFromFile(
      filename.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
  //, *_network_definition, DataType::kFLOAT);
  if (!success) {
    _last_error = "Can't parse model file " +
                  std::string(_onnx_parser->getError(0)->desc());
    return false;
  }

  auto prof = _builder->createOptimizationProfile();

  prof->setDimensions(
      getInputNodeName().c_str(), OptProfileSelector::kMIN,
      nvinfer1::Dims4{(int)_batch_size, getInputHeight(), getInputWidth(), 3});
  prof->setDimensions(
      getInputNodeName().c_str(), OptProfileSelector::kMAX,
      nvinfer1::Dims4{(int)_batch_size, getInputHeight(), getInputWidth(), 3});
  prof->setDimensions(
      getInputNodeName().c_str(), OptProfileSelector::kOPT,
      nvinfer1::Dims4{(int)_batch_size, getInputHeight(), getInputWidth(), 3});

  _builder_config->addOptimizationProfile(prof);

  _cuda_engine = samplesCommon::InferObject(
      _builder->buildEngineWithConfig(*_network_definition, *_builder_config));

  if (!_cuda_engine) {
    _last_error = "Can't build cuda engine";
    return false;
  }

  _context = samplesCommon::InferObject(_cuda_engine->createExecutionContext());

  if (!_context) {
    _last_error = "Can't create context";
    return false;
  }

  _context->setBindingDimensions(
      0,
      nvinfer1::Dims4{(int)_batch_size, getInputHeight(), getInputWidth(), 3});

  // Create RAII buffer manager object
  _buffers = std::shared_ptr<samplesCommon::BufferManager>(
      new samplesCommon::BufferManager(_cuda_engine, (int)_batch_size));

  _model_filename = filename;

  _is_loaded = true;
  return true;
}

bool TRTCNNInferencer::loadFromCudaEngine(const string &filename) {
  _is_loaded = false;

  if (filename.rfind(".ce") == std::string::npos &&
      filename.rfind(".bin") == std::string::npos) {
    _last_error = "File is not Cuda Engine binary model";
    return false;
  }

  if (!FileExists(filename)) {
    _last_error = "File not exists";
    return false;
  }

  clearModel();

  _builder->setMaxBatchSize(_batch_size);

//  configureBitMode();
//  configureGPUMemory();

  std::ifstream input(filename, std::ios::binary);
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

  _cuda_engine = samplesCommon::InferObject(
      _runtime->deserializeCudaEngine(buffer.data(), buffer.size()));

  if (!_cuda_engine) {
    _last_error = "Can't build cuda engine";

    return false;
  }

  _context = samplesCommon::InferObject(_cuda_engine->createExecutionContext());

  if (!_context) {
    _last_error = "Can't create context";
    return false;
  }

  // Create RAII buffer manager object
  _buffers = std::shared_ptr<samplesCommon::BufferManager>(
      new samplesCommon::BufferManager(_cuda_engine, _batch_size));

  _model_filename = filename;

  _is_loaded = true;
  return true;
}

void TRTCNNInferencer::configureBitMode() {

  if (!_builder) {
    return;
  }
  if (!_builder_config) {
    return;
  }

  _builder_config->clearFlag(nvinfer1::BuilderFlag::kINT8);
  _builder_config->clearFlag(nvinfer1::BuilderFlag::kFP16);

  if (_bit_mode == BIT_MODE::AUTO) {
    if (_builder->platformHasFastInt8()) {
      _bit_mode = BIT_MODE::INT8;
    } else if (_builder->platformHasFastFp16()) {
      _bit_mode = BIT_MODE::FLOAT16;
    } else {
      _bit_mode = BIT_MODE::FLOAT32;
    }
    return configureBitMode();
  }

  else if (_bit_mode == BIT_MODE::FLOAT16) {
    _builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if (_bit_mode == BIT_MODE::INT8) {
    _builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
  } else if (_bit_mode == BIT_MODE::FLOAT32) {
    // nothing, flags were skipped
  }
}

bool TRTCNNInferencer::loadFromUff(const string &filename) {
  _is_loaded = false;

  if (filename.rfind(".uff") == std::string::npos) {
    _last_error = "File is not UFF model";
    return false;
  }

  if (!FileExists(filename)) {
    _last_error = "File not exists";
    return false;
  }

  clearModel();

  _uff_parser->registerInput(_input_node_name.c_str(), _input_shape,
                             nvuffparser::UffInputOrder::kNHWC);

  for (size_t i = 0; i < _output_node_names.size(); i++)
    _uff_parser->registerOutput(_output_node_names[i].c_str());

  _builder->setMaxBatchSize((int)_batch_size);
  configureGPUMemory();
  configureBitMode();

  bool success = _uff_parser->parse(filename.c_str(), *_network_definition,
                                    DataType::kFLOAT);
  if (!success) {
    _last_error = "Can't parse model file";
    return false;
  }

  _cuda_engine = samplesCommon::InferObject(
      _builder->buildEngineWithConfig(*_network_definition, *_builder_config));

  if (!_cuda_engine) {
    _last_error = "Can't build cuda engine";
    return false;
  }

  _context = samplesCommon::InferObject(_cuda_engine->createExecutionContext());

  if (!_context) {
    _last_error = "Can't create context";
    return false;
  }

  // Create RAII buffer manager object
  _buffers = std::shared_ptr<samplesCommon::BufferManager>(
      new samplesCommon::BufferManager(_cuda_engine, _batch_size));

  _model_filename = filename;

  _is_loaded = true;
  return true;
}

std::string TRTCNNInferencer::inference(const std::vector<cv::Mat> &imgs) {
  if (!_cuda_engine) {
    _last_error = "Not loaded model";
    return _last_error;
  }

  if (!_context) {
    _last_error = "Can't create context";
    return _last_error;
  }

#ifdef TRT_DEBUG
  auto t1 = std::chrono::high_resolution_clock::now();
#endif

  bool success =
      processInput(*_buffers, imgs, {0.485, 0.456, 0.406,}, _norm_type, _bgr2rgb);
  _buffers->copyInputToDevice();

  if (!success) {
    return _last_error;
  }

  // NOTE: Execute V2?
  // "Current optimization profile is: 0." ??? Is it good? How to change?
  bool status =
      _context->execute(_batch_size, _buffers->getDeviceBindings().data());

  if (!status) {
    _last_error = "Can't execute context";
    return _last_error;
  }

  // Memcpy from device output buffers to host output buffers
  _buffers->copyOutputToHost();
  // WARN: and here we go to parse...

#ifdef TRT_DEBUG
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Inference took: " << duration << " microseconds" << std::endl;
#endif

  return "OK";
}

int TRTCNNInferencer::getInputHeight() const { return _input_shape.d[0]; }

void TRTCNNInferencer::setInputHeight(const int &height) {
  _input_shape.d[0] = height;
}

int TRTCNNInferencer::getInputWidth() const { return _input_shape.d[1]; }

void TRTCNNInferencer::setInputWidth(const int &width) {
  _input_shape.d[1] = width;
}

int TRTCNNInferencer::getInputDepth() const { return _input_shape.d[2]; }

void TRTCNNInferencer::setInputDepth(const int &depth) {
  _input_shape.d[2] = depth;
}

std::string TRTCNNInferencer::getLastError() const { return _last_error; }

std::string TRTCNNInferencer::getModelFilename() const {
  return _model_filename;
}

TRTCNNInferencer::BIT_MODE TRTCNNInferencer::getBitMode() const {
  return _bit_mode;
}

void TRTCNNInferencer::setBitMode(const BIT_MODE &bit_mode) {
  _bit_mode = bit_mode;
}

bool TRTCNNInferencer::processInput(const samplesCommon::BufferManager &buffers,
                                    const std::vector<cv::Mat> &imgs,
                                    const std::vector<float> &mean,
                                    NormalizeType normalize, bool rgb) {

  const int inputC = getInputDepth();
  const int inputH = getInputHeight();
  const int inputW = getInputWidth();

  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(_input_node_name));
  // NOTE: Carefully, size is in bytes!
  const int size = buffers.size(_input_node_name) / sizeof(float);

  if (!hostDataBuffer) {
    _last_error = "Can not get input tensor by name: " + _input_node_name;
    return false;
  }

  // Host memory for input buffer
  const int volImg = inputC * inputH * inputW;
  const int volChl = inputH * inputW;
  cv::Mat input_img;

  for (int i = 0; i < _batch_size; ++i) {

    // Nearest is faster, but results are different
    cv::resize(imgs[i], input_img, cv::Size(inputW, inputH), 0, 0);

    // positions - height - width
    input_img.forEach<cv::Vec3b>([&](cv::Vec3b &pixel,
                                     const int position[]) -> void {
      for (short c = 0; c < inputC; ++c) {
        float val(pixel[c]);
        val /= 255.0;
        val -= mean[2-c];
        val /= deviation[2-c];

//        if (normalize == NormalizeType::DETECTION) {
//          val = (2.0f / 255.0f) * val - 1.0f;
//        } else if (normalize == NormalizeType::CLASSIFICATION_SLIM) {
//          // WARN: IDK why, this shouldnt happen, but work only with this
//          // preprocessing
//          val = float(val);
//          val /= 255.0;
//          // val -= 0.5;
//          // val *= 2.0;
//        }

        int pos = 0;
        if (rgb) {
          pos = i * volImg + (2 - c) * volChl + position[0] * inputW + position[1];
        } else {
          pos = i * volImg + c * volChl + position[0] * inputH + position[1];
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
  _builder->reset();
  _builder_config->reset();
}

bool TRTCNNInferencer::getBGR2RGBConvertionEnabled() const { return _bgr2rgb; }

void TRTCNNInferencer::setBGR2RGBConvertionEnabled(bool bgr2rgb) {
  _bgr2rgb = bgr2rgb;
}

NormalizeType TRTCNNInferencer::getNormalizationType() const {
  return _norm_type;
}

void TRTCNNInferencer::setNormalizationType(const NormalizeType &norm_type) {
  _norm_type = norm_type;
}
