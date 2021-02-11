#include "segmentation_wrapper.h"
#include "gtest/gtest.h"

#include <opencv2/opencv.hpp>

bool check_equality(const cv::Mat &a, const cv::Mat &b) {
  return std::equal(a.begin<uchar>(), a.end<uchar>(), b.begin<uchar>());
}

template <typename T> std::vector<T> convertMatToVec(cv::Mat &mat) {
  std::vector<T> v;
  v.assign((T *)mat.datastart, (T *)mat.dataend);
  return v;
}

class TestSegmentation : public ::testing::Test {
protected:
  void SetUp() override {
    img = cv::imread("images/left_2_000000318.jpg");
    index_mask_actual =
        cv::imread("images/1_trt_index.png", cv::IMREAD_GRAYSCALE);
    color_mask_actual = cv::imread("images/1_trt_color.png");
    seg_wrapper.prepareForInference("config.json");
  }
  SegmentationWrapper seg_wrapper;
  cv::Mat img;
  cv::Mat index_mask_actual;
  cv::Mat color_mask_actual;

  void TearDown() override {}
};

TEST_F(TestSegmentation, index_and_mask) {
  seg_wrapper.inference(img);

  // Sanity check
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_actual));
  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_actual));

  cv::Mat index_mask_pred = seg_wrapper.getIndexMask();
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_pred));
  cv::Mat color_mask_pred = seg_wrapper.getColorMask(0.4, img);
  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_pred));
  color_mask_pred = seg_wrapper.getColorMask(0.4, img);
  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_pred));
  index_mask_pred = seg_wrapper.getIndexMask();
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_pred));
}

TEST_F(TestSegmentation, mask_and_index) {
  seg_wrapper.inference(img);

  cv::Mat color_mask_pred = seg_wrapper.getColorMask(0.4, img);
  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_pred));
  cv::Mat index_mask_pred = seg_wrapper.getIndexMask();
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_pred));
  index_mask_pred = seg_wrapper.getIndexMask();
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_pred));
  color_mask_pred = seg_wrapper.getColorMask(0.4, img);
  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_pred));
}

TEST_F(TestSegmentation, index_multiple) {
  seg_wrapper.inference(img);

  cv::Mat index_mask_pred = seg_wrapper.getIndexMask();
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_pred));
  index_mask_pred = seg_wrapper.getIndexMask();
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_pred));
  index_mask_pred = seg_wrapper.getIndexMask();
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_pred));
}

TEST_F(TestSegmentation, mask_multiple) {
  seg_wrapper.inference(img);

  cv::Mat color_mask_pred = seg_wrapper.getColorMask(0.4, img);
  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_pred));
  color_mask_pred = seg_wrapper.getColorMask(0.4, img);
  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_pred));
  color_mask_pred = seg_wrapper.getColorMask(0.4, img);
  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_pred));
}

TEST_F(TestSegmentation, infer_multiple_images) {
  seg_wrapper.inference(img);

  cv::Mat color_mask_pred = seg_wrapper.getColorMask(0.4, img);
  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_pred));
  cv::Mat index_mask_pred = seg_wrapper.getIndexMask();
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_pred));

  cv::Mat img2 = cv::imread("images/left_1_000000222.jpg");
  cv::Mat color_mask_actual2 = cv::imread("images/2_trt_color.png");
  cv::Mat index_mask_actual2 = cv::imread("images/2_trt_index.png");
  seg_wrapper.inference(img2);

  cv::Mat color_mask_pred2 = seg_wrapper.getColorMask(0.4, img2);
  ASSERT_TRUE(check_equality(color_mask_actual2, color_mask_pred2));
  cv::Mat index_mask_pred2 = seg_wrapper.getIndexMask();
  ASSERT_TRUE(check_equality(index_mask_actual2, index_mask_pred2));
}

TEST_F(TestSegmentation, index_and_mask_postprocess) {
  cv::Mat img2 = cv::imread("images/left_1_000000222.jpg");
  cv::Mat color_mask_actual2 = cv::imread("images/2_trt_color_postprocessed.png");
  cv::Mat index_mask_actual2 = cv::imread("images/2_trt_index_postprocessed.png");
  seg_wrapper.inference(img2);

  cv::Mat index_mask_pred = seg_wrapper.getIndexMask(200);
  ASSERT_TRUE(check_equality(index_mask_actual2, index_mask_pred));
  cv::Mat color_mask_pred = seg_wrapper.getColorMask(0.4, img2, 200);
  ASSERT_TRUE(check_equality(color_mask_actual2, color_mask_pred));
}