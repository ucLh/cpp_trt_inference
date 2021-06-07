#include "trt_yolo_inferencer.h"

cv::Rect2f TRTYoloInferencer::processBox(float xmin, float ymin, float xmax,
                                         float ymax, int index) {
  const float x = xmin * m_current_original_cols[index];
  const float y = ymin * m_current_original_rows[index];
  const float x1 = xmax * m_current_original_cols[index];
  const float y1 = ymax * m_current_original_rows[index];
  return cv::Rect2f(x, y, x1 - x, y1 - y);
}

TRTYoloInferencer::TRTYoloInferencer() {
  m_norm_type = NormalizeType::DETECTION_YOLOV4;
}
