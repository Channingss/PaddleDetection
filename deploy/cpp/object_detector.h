//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils/yml_parse.h"
#include "preprocess/preprocess_ops.h"
#include "paddle_inference_api.h" // NOLINT


using namespace std;
namespace PaddleDetection {
// Object Detection Result
struct ObjectResult {
  // Rectangle coordinates of detected object: left, right, top, down
  std::vector<int> rect;
  // Class id of detected object
  int class_id;
  // Confidence of detected object
  float confidence;
};

// Model Configuration
struct ModelConfig {
  bool is_scale;
  bool to_bgr;

  int eval_max_size;
  int eval_target_size;
  int padding_stride;

  std::vector<float> mean;
  std::vector<float> scale;
};


// Visualiztion MaskDetector results
void VisualizeResult(const cv::Mat& img,
                     const std::vector<ObjectResult>& results,
                     cv::Mat* vis_img);

class ObjectDetector {
 public:
  explicit ObjectDetector(const std::string& model_dir,
                          const utils::ConfigPaser& config,
                          bool use_gpu = false,
                          float threshold = 0.7) :
      threshold_(threshold) {
    config_ = config;
    preprocess::ParsePreprocessInfo(config_.preprocess_info, config_.arch, &preprocess_ops);
    LoadModel(model_dir, use_gpu);
  }

  // Load Paddle inference model
  void LoadModel(
    const std::string& model_dir,
    bool use_gpu);

  // Run predictor
  void Predict(
      const cv::Mat& img,
      std::vector<ObjectResult>* result);

 private:
  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& image_mat);
  // Postprocess result
  void Postprocess(
      const cv::Mat& raw_mat,
      std::vector<ObjectResult>* result);

  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  preprocess::PreprocessOps preprocess_ops;
  std::vector<float> input_data_;
  std::vector<float> output_data_;
  std::vector<int> input_shape_;
  float threshold_;
  utils::ConfigPaser config_;
};

}  // namespace PaddleDetection
