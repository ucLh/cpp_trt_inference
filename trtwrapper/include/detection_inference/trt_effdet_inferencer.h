#ifndef TRT_INFERENCE_TRT_EFFDET_INFERENCER_H
#define TRT_INFERENCE_TRT_EFFDET_INFERENCER_H

#include "trt_detection_inferencer.h"

class TRTEffdetInferencer : public TRTDetectionInferencer {
public:
  TRTEffdetInferencer();
  cv::Rect2f processBox(float xmin, float ymin, float xmax, float ymax,
                        int index) override;
protected:
  int remapClassIndex(int cl_index) override;
};

#endif // TRT_INFERENCE_TRT_EFFDET_INFERENCER_H
