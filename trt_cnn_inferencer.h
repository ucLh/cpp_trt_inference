#ifndef TRT_CNN_INFERENCER_H
#define TRT_CNN_INFERENCER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include "logger.h"

#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <ostream>
#include <iterator>
#include "trt_common.hpp"
#include "trt_buffers.hpp"

#include <chrono>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#define TRT_DEBUG


template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

enum NormalizeType {
    DETECTION,
    CLASSIFICATION_SLIM
};

///
/// \brief The TRTCNNInferencer class defines interface for TRT inferencers: Detection, Classification, Segmentation, etc
/// It's verrritual class do not use it.
///
class TRTCNNInferencer
{
public:

    ///
    /// \brief The BIT_MODE enum
    /// FP16 is faster than FFP32 but accuracy can be UNPREDICTABLE changed. Supposed that not much.
    /// INT8 even more faster but accuracyy drain will bee huge for  sure.
    /// Please note, for FP16 and INT8 hardware must be implemented. If you  not sure set AUTO for max speed and
    /// if special bits available they will be used (int* biggest priority on AUTO)).
    ///
    enum BIT_MODE: int {
        FLOAT32 = 0, // Default
        FLOAT16 = 1,
        INT8 = 2,
        AUTO = 3 // Will be replaced with most optimal flag during building model
    };

    TRTCNNInferencer();
    virtual ~TRTCNNInferencer();

    // No reason to support copy constructor
    TRTCNNInferencer(const TRTCNNInferencer&) = delete;
    TRTCNNInferencer(TRTCNNInferencer&& that) = default;


    ///
    /// \brief loadFromONNX on 22.04.2020 no one detection model loaded with it or UFF successful (classification only works)
    /// but potentially it's the best way to load model. Therefore on each device
    /// best optimization to cuda engine format will be done on load stage.
    /// If one day this problem will be  solved: https://github.com/onnx/onnx-tensorrt/issues/401#issuecomment-615203619
    /// therefore maybe I will be ablke to work with ONNX models.
    /// Also, during load of exported ONNX model you can face with this: https://github.com/onnx/onnx-tensorrt/issues/400#issuecomment-601123753
    /// I workarounded it with onnx-tensorrt source modifications (use int8 instead of uint8 is safe in ranges of network typical weights most probable).
    /// \param filename
    /// \return true if all was ok. If not check getLastError method
    ///
    virtual bool loadFromONNX(const std::string& filename);

    ///
    /// \brief loadFromCudaEngine the only method that works for now. Can be loaded only on the same GPU or very similar
    /// Otherwise program can crash. Export now  works only with TRT on Python: https://github.com/AastaNV/TRT_object_detection
    /// \param filename
    /// \return true if all was ok. If not check getLastError method
    ///
    virtual bool loadFromCudaEngine(const std::string& filename);

    ///
    /// \brief loadFromUff see loadFromONNX
    /// \param filename
    /// \return true if all was ok. If not check getLastError method
    ///
    virtual bool loadFromUff(const std::string& filename);

    ///
    /// \brief isLoaded
    /// \return  true if last load successful. If not check getLastError method
    ///
    virtual inline bool isLoaded() const { return _is_loaded; }

    // just name for helps, nothing functional
    virtual inline std::string getName() const { return _name; }
    virtual void setName(const std::string& name) { _name = name; }

    ///
    /// \brief warmUp
    /// warmUpializes resources and prepares model for fast inference. Not necessary to call it
    /// but otherwise first model inference call can be slow.
    ///
    virtual void warmUp() {};

    ///
    /// \brief getGpuMemoryFraction from 0.0 to 1.0, means percents of GPU memory will be used.
    /// \return
    ///
    double getGpuMemoryFraction() const {
        return _gpu_memory_fraction;
    }
    void setGpuMemoryFraction(double gpu_memory_fraction) {
        _gpu_memory_fraction = gpu_memory_fraction;
    }


    ///
    /// \brief inference
    /// \param imgs
    /// \return status string, if all o will be "OK" or empty
    ///  If not check getLastError method
    virtual std::string inference(const std::vector<cv::Mat> &imgs);

    std::string getInputNodeName() const { return _input_node_name; }
    void setInputNodeName(const std::string &name) {
        _input_node_name = name;
    }

    std::vector<std::string> getOutputNodeName() const {
        return _output_node_names;
    }
    void setOutputNodeName(const std::vector<std::string>& output_node_names) {
        _output_node_names = output_node_names;
    }

    int getInputHeight() const;
    void setInputHeight(const int &height);

    int getInputWidth() const;
    void setInputWidth(const int &width);

    int getInputDepth() const;
    void setInputDepth(const int &depth);

    std::vector<std::string> getLabelNames() const;
    void setLabelNames(const std::vector<std::string> &label_names);

    std::vector<std::vector<int> > getLabelColours() const;
    void setLabelColours(const std::vector<std::vector<int> > &label_colours);

    std::string getLastError() const;

    std::string getModelFilename() const;

    BIT_MODE getBitMode() const;
    void setBitMode(const BIT_MODE &bit_mode);

    NormalizeType getNormalizationType() const;
    void setNormalizationType(const NormalizeType &norm_type);

    bool getBGR2RGBConvertionEnabled() const;
    void setBGR2RGBConvertionEnabled(bool bgr2rgb);

protected:

    void configureBitMode();

    virtual bool processInput(const samplesCommon::BufferManager &buffers,
                      const std::vector<cv::Mat>& imgs,
                      const std::vector<float>& mean = {0, 0, 0},
                      NormalizeType normalize = NormalizeType::DETECTION, bool rgb = true);

    virtual bool processOutput(const samplesCommon::BufferManager &buffers) = 0;

    virtual void clearModel();
    void parseName(const std::string& filename);
    void configureGraph();
    void configureGPUMemory();

    std::string _last_error = "";
    std::string _model_filename = "";
    BIT_MODE _bit_mode = BIT_MODE::FLOAT16;

    std::shared_ptr<nvinfer1::IBuilder> _builder = nullptr;

    std::shared_ptr<nvonnxparser::IParser> _onnx_parser = nullptr;

    std::shared_ptr<nvuffparser::IUffParser> _uff_parser = nullptr;

    std::shared_ptr<nvinfer1::INetworkDefinition> _network_definition = nullptr;

    nvinfer1::Dims3 _input_shape = {3, 300, 300};

    //nvuffparser::UffInputOrder _input_order = nvuffparser::UffInputOrder::kNCHW;
    std::shared_ptr<nvinfer1::IBuilderConfig> _builder_config = nullptr;
    std::shared_ptr<nvinfer1::ICudaEngine> _cuda_engine{nullptr};
    std::shared_ptr<IRuntime> _runtime = nullptr;
    std::shared_ptr<IExecutionContext> _context = nullptr;

    std::shared_ptr<samplesCommon::BufferManager> _buffers;

    size_t _batch_size = 1;

    std::vector<std::vector<int>> _label_colors = {
                                            {0, 0, 0}
    };

    std::vector<std::string> _label_names = {
        "none"
    };


    std::string _input_node_name = "Input";
    std::vector<std::string> _output_node_names = {"NMS"};



    bool _is_loaded = false;

    std::string _name = "UnknownModel";
    double _gpu_memory_fraction = 1.0;

    NormalizeType _norm_type = NormalizeType::DETECTION;
    bool _bgr2rgb = true; // otherwise RGB
};

#endif // TRT_CNN_INFERENCER_H
