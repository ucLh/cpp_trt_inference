#include "detection_inference/trt_effdet_inferencer.h"

cv::Rect2f TRTEffdetInferencer::processBox(float xmin, float ymin,
                                              float xmax, float ymax,
                                              int index) {
  const float x = xmin / ((float)m_cols / m_current_original_cols[index]);
  const float y = ymin / ((float)m_rows / m_current_original_rows[index]);
  const float x1 = xmax / ((float)m_cols / m_current_original_cols[index]);
  const float y1 = ymax / ((float)m_rows / m_current_original_rows[index]);
  return cv::Rect2f(x, y, x1 - x, y1 - y);
}

TRTEffdetInferencer::TRTEffdetInferencer() {
  m_norm_type = NormalizeType::DETECTION_EFFDET;
}