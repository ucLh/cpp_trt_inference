#include "trt_cnn_inferencer.h"

#include <memory>

TRTCNNInferencer::TRTCNNInferencer() {
  m_builder = samplesCommon::InferObject(nvinfer1::createInferBuilder(gLogger));
  m_builder_config =
      samplesCommon::InferObject(m_builder->createBuilderConfig());

  m_runtime = samplesCommon::InferObject(createInferRuntime(gLogger));

  initLibNvInferPlugins(&gLogger, "");
}

TRTCNNInferencer::~TRTCNNInferencer() = default;

bool TRTCNNInferencer::loadFromCudaEngine(const string &filename) {
//  m_is_loaded = false;

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

  std::ifstream input(filename, std::ios::binary);
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

  m_cuda_engine = samplesCommon::InferObject(
      m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));

  if (!m_cuda_engine) {
    m_last_error = "Can't build cuda engine";

    return false;
  }

  m_context =
      samplesCommon::InferObject(m_cuda_engine->createExecutionContext());

  if (!m_context) {
    m_last_error = "Can't create context";
    return false;
  }

  // Create RAII buffer manager object
  m_buffers = std::make_shared<samplesCommon::BufferManager>(m_cuda_engine,
                                                             m_batch_size);

  //  m_is_loaded = true;
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

  bool success = processInput(*m_buffers, imgs,
                              {
                                  0.485,
                                  0.456,
                                  0.406,
                              },
                              m_norm_type, m_bgr2rgb);
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

#ifdef TRT_DEBUG
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Inference took: " << duration << " microseconds" << std::endl;
#endif

  return "OK";
}

int TRTCNNInferencer::getInputHeight() const { return m_input_shape.d[0]; }

int TRTCNNInferencer::getInputWidth() const { return m_input_shape.d[1]; }

int TRTCNNInferencer::getInputDepth() const { return m_input_shape.d[2]; }

std::string TRTCNNInferencer::getLastError() const { return m_last_error; }

bool TRTCNNInferencer::processInput(const samplesCommon::BufferManager &buffers,
                                    const std::vector<cv::Mat> &imgs,
                                    const std::vector<float> &mean,
                                    NormalizeType normalize, bool rgb) {

  const int inputC = getInputDepth();
  const int inputH = getInputHeight();
  const int inputW = getInputWidth();

  auto *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(m_input_node_name));

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
        if ((normalize == NormalizeType::SEGMENTATION) ||
            (normalize == NormalizeType::DETECTION_EFFDET)) {
          val /= 255.0;
          val -= mean[2 - c];
          val /= m_deviation[2 - c];
        } else if (normalize == NormalizeType::DETECTION) {
          val = (2.0f / 255.0f) * val - 1.0f;
        } else if (normalize == NormalizeType::CLASSIFICATION_SLIM) {
          // WARN: IDK why, this shouldnt happen, but work only with this
          // preprocessing
          val = float(val);
          val /= 255.0;
          // val -= 0.5;
          // val *= 2.0;
        } else if (normalize == NormalizeType::DETECTION_YOLOV4) {
          val /= 255.0;
        }

        int pos = 0;
        if (rgb) {
          pos = i * volImg + (2 - c) * volChl + position[0] * inputW +
                position[1];
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
  for (const auto &e : check_)
    outFile << e << "\n";
#endif

  return true;
}

void TRTCNNInferencer::clearModel() {
  m_builder->reset();
  m_builder_config->reset();
}
