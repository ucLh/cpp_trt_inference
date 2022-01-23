#ifndef TRT_INFERENCE_TRT_YOLO_INFERENCER_H
#define TRT_INFERENCE_TRT_YOLO_INFERENCER_H

#include "trt_detection_inferencer.h"

class TRTYoloInferencer : public TRTDetectionInferencer {
public:
  TRTYoloInferencer();
  cv::Rect2f processBox(float xmin, float ymin, float xmax, float ymax,
                        int index) override;
protected:
  int remapClassIndex(int cl_index) override;
};

#endif // TRT_INFERENCE_TRT_YOLO_INFERENCER_H
