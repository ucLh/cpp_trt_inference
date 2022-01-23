#include "detection_inference/trt_effdet_inferencer.h"

TRTEffdetInferencer::TRTEffdetInferencer() {
  m_norm_type = NormalizeType::DETECTION_EFFDET;
}

cv::Rect2f TRTEffdetInferencer::processBox(float xmin, float ymin,
                                              float xmax, float ymax,
                                              int index) {
  const float x = xmin / ((float)m_cols / m_current_original_cols[index]);
  const float y = ymin / ((float)m_rows / m_current_original_rows[index]);
  const float x1 = xmax / ((float)m_cols / m_current_original_cols[index]);
  const float y1 = ymax / ((float)m_rows / m_current_original_rows[index]);
  return cv::Rect2f(x, y, x1 - x, y1 - y);
}

int TRTEffdetInferencer::remapClassIndex(int cl_index) {
  int final_cl_index = cl_index;
  if ((1 <= cl_index) && (cl_index < 9)) {
    // Vehicles
    final_cl_index = 1;
  } else if ((9 <= cl_index) && (cl_index < 15)) {
    // Pillar
    final_cl_index = 3;
  } else if ((14 <= cl_index) && (cl_index < 25)) {
    // Animals
    final_cl_index = 2;
  } else if (cl_index != 0) {
    // Other, not person
    final_cl_index = 4;
  }
  return final_cl_index;
}