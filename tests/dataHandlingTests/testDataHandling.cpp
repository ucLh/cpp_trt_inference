//#include "segmentation_wrapper.h"
#include "gtest/gtest.h"

#include "trt_segmentation_inferencer.h"

class TestTRTSegmentationInferencer : public TRTSegmentationInferencer {
public:
  IDataBase::ConfigData getDataHandlerConfig() {
    IDataBase::ConfigData config;
    config.colors_path = m_data_handler->getConfigColorsPath();
    config.engine_path = m_data_handler->getConfigEnginePath();
    config.output_nodes = m_data_handler->getConfigOutputNodes();
    config.input_node = m_data_handler->getConfigInputNode();
    config.input_size = m_data_handler->getConfigInputSize();
    return config;
  }
};

TEST(TestDataHandling, config_sets_up) {
  TestTRTSegmentationInferencer inferencer;
  IDataBase::ConfigData config_manual;
  config_manual = {
      cv::Size(60, 128),
      "int_0",
      {"outt_0"},
      "/home/luch/effnetb0_unet_gray_2grass_iou55_640x120_argmax.bin",
      "/home/luch/Programming/C++/cpp_trt_inference/classes.csv",
      ""};

  inferencer.prepareForInference(config_manual);
  auto config_from_inferecer = inferencer.getDataHandlerConfig();
  ASSERT_EQ(config_manual.input_size, config_from_inferecer.input_size);
  ASSERT_EQ(config_manual.input_node, config_from_inferecer.input_node);
  ASSERT_EQ(config_manual.output_nodes, config_from_inferecer.output_nodes);
  ASSERT_EQ(config_manual.engine_path, config_from_inferecer.engine_path);
  ASSERT_EQ(config_manual.colors_path, config_from_inferecer.colors_path);
  ASSERT_EQ(config_manual.detection_labels_path,
            config_from_inferecer.detection_labels_path);
}
// TODO: Add the same test for detection