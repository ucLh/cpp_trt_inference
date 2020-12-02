#ifndef TRT_INFERENCE_UTIL_H
#define TRT_INFERENCE_UTIL_H

#include <opencv2/opencv.hpp>

bool smart_resize(const cv::Mat &in, cv::Mat &dist, const cv::Size &size, int interpolation = cv::INTER_LINEAR) {
  if (in.size() == size) {
    in.copyTo(dist);
    return true;
  }

  cv::resize(in, dist, size, 0, 0, interpolation);
  return true;
}

#endif // TRT_INFERENCE_UTIL_H
