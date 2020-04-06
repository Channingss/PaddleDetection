#include <iostream>
#include <string>

#include "object_detector.h"

int main(int argc, char** argv)
{
  std::string model_dir(argv[1]);
  std::string image_file(argv[2]);
  int use_gpu = std::stoi(argv[3]);
  PaddleDetection::utils::ConfigPaser config;
  config.load_config(model_dir);
  PaddleDetection::ObjectDetector det(model_dir, config, use_gpu);

  cv::Mat im = cv::imread(image_file, -1);
  vector<PaddleDetection::ObjectResult> result;
  det.Predict(im, &result);
  return 0;
}
