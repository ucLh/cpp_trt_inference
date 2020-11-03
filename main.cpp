#include <iostream>

#include <setjmp.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <vector>

//#include "tensorflow/core/public/session.h"

// #include "opencv2/opencv.hpp"
#include <opencv2/opencv.hpp>

//#include "tensorflow_utils.hpp"
//#include "tensorflow_segmentator.hpp"
//#include "tensorflow_classifier.hpp"
//#include "tensorflow_detector.hpp"
#include "trt_detection_inferencer.h"
#include "trt_classification_inferencer.h"



using namespace std;
//using namespace tensorflow;
//using namespace tf_utils;

#include <iostream>

int main()
{


    /// Tensorflow


//     {
//         TensorflowDetector inferencer_tf;
//         inferencer_tf.setGpuMemoryFraction(0.2);

//         inferencer_tf.load("auto_v4_trt.pb");
//         inferencer_tf.warmUp();

//         auto img = cv::imread("1.jpg");
//         inferencer_tf.inference({img});
//         cv::Mat im = inferencer_tf.getFramesWithBoundingBoxes(0.5)[0];
//         cv::imwrite("1_tf_.jpg", im);

//        cv::waitKey(0);

// //        auto t1 = std::chrono::high_resolution_clock::now();

// //        std::cout << "Starting inference TF..." << std::endl;
// //        for (int i = 0; i < 10000; ++i)
// //            inferencer_tf.inference({img});

// //        auto t2 = std::chrono::high_resolution_clock::now();
// //        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
// //        std::cout << "Inference TF took: " << (duration / 10000) << " microseconds" <<
// //  std::endl;
//     }





    /// TensortRT


   {
       cv::Mat img = cv::imread("1.jpg");

       TRTDetectionInferencer inferencer;
       bool ok = inferencer.loadFromCudaEngine("auto_v4_trt_desktop.bin");
       //bool ok = inferencer.loadFromUff("ssd_mobilenet_v1_coco.uff");
       std::cerr << "Status of load: " << inferencer.getLastError() << std::endl;

       inferencer.inference({img});
       std::cerr << "Status of inference: " << inferencer.getLastError() << std::endl;
       std::cout << "Size:  " << inferencer.getFramesWithBoundingBoxes().size() << std::endl;
       cv::Mat im = inferencer.getFramesWithBoundingBoxes()[0];
       cv::imwrite("1_trt_.jpg", im);
       cv::waitKey(0);

       auto t1 = std::chrono::high_resolution_clock::now();

       std::cout << "Starting inference TRT..." << std::endl;
       for (int i = 0; i < 10000; ++i)
           inferencer.inference({img});

       auto t2 = std::chrono::high_resolution_clock::now();
       auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
       std::cout << "Inference TensorRT took: " << (duration / 10000) << " microseconds" << std::endl;
   }

    return 0;
}
