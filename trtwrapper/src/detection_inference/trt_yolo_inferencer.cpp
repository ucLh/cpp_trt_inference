#include "detection_inference/trt_yolo_inferencer.h"

TRTYoloInferencer::TRTYoloInferencer() {
  m_norm_type = NormalizeType::DETECTION_YOLOV4;
}

cv::Rect2f TRTYoloInferencer::processBox(float xmin, float ymin, float xmax,
                                         float ymax, int index) {
  const float x = xmin * m_current_original_cols[index];
  const float y = ymin * m_current_original_rows[index];
  const float x1 = xmax * m_current_original_cols[index];
  const float y1 = ymax * m_current_original_rows[index];
  return cv::Rect2f(x, y, x1 - x, y1 - y);
}

int TRTYoloInferencer::remapClassIndex(int cl_index) {
  int final_cl_index = cl_index;
  if ((1 <= cl_index) && (cl_index < 9)) {
    // Vehicles
    final_cl_index = 1;
  } else if ((9 <= cl_index) && (cl_index < 14)) {
    // Pillar
    final_cl_index = 3;
  } else if ((14 <= cl_index) && (cl_index < 24)) {
    // Animals
    final_cl_index = 2;
  } else if (cl_index != 0) {
    // Other, not person
    final_cl_index = 4;
  }
  return final_cl_index;
}
