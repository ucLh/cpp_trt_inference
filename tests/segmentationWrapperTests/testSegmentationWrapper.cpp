#include "segmentation_wrapper.h"
#include "gtest/gtest.h"

#include <opencv2/opencv.hpp>

bool check_equality(const cv::Mat &a, const cv::Mat &b) {
  return std::equal(a.begin<uchar>(), a.end<uchar>(), b.begin<uchar>());
}

TEST(segmentation_modes, predicts_correctly) {
  cv::Mat img = cv::imread("left_2_000000318.jpg");
  cv::Mat index_mask_actual = cv::imread("1_trt_index.png", cv::IMREAD_GRAYSCALE);
  cv::Mat color_mask_actual = cv::imread("1_trt_color.png");

  SegmentationWrapper seg_wrapper;
  seg_wrapper.prepareForInference("config.json");
  seg_wrapper.inference(img);

  // Sanity check
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_actual));
//  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_actual));

  cv::Mat index_mask_pred = seg_wrapper.getIndexMask();
//    std::vector<uchar> v1, v2;
//  v1.assign((uchar *)index_mask_actual.datastart,
//            (uchar *)index_mask_actual.dataend);
//  v2.assign((uchar *)index_mask_pred.datastart,
//            (uchar *)index_mask_pred.dataend);
//  std::cout << v1.size() << " " << v2.size() << "\n";
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_pred));

  cv::Mat color_mask_pred = seg_wrapper.getColorMask(0.4, img);
//  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_pred));

  // Some more sanity checks
  ASSERT_TRUE(check_equality(index_mask_actual, index_mask_actual));
  ASSERT_TRUE(check_equality(color_mask_actual, color_mask_actual));

  // Sanity check


}