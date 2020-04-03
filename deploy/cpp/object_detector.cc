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


# include "object_detector.h"

// Normalize the image by (pix - mean) * scale
void NormalizeImage(
    const std::vector<float> &mean,
    const std::vector<float> &scale,
    cv::Mat& im, // NOLINT
    float* input_buffer) {
  int height = im.rows;
  int width = im.cols;
  int stride = width * height;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int base = h * width + w;
      input_buffer[base + 0 * stride] =
          (im.at<cv::Vec3f>(h, w)[0] - mean[0]) * scale[0];
      input_buffer[base + 1 * stride] =
          (im.at<cv::Vec3f>(h, w)[1] - mean[1]) * scale[1];
      input_buffer[base + 2 * stride] =
          (im.at<cv::Vec3f>(h, w)[2] - mean[2]) * scale[2];
    }
  }
}

// Load Model and create model predictor void Object::LoadModel(const std::string& model_dir, bool use_gpu) {
void ObjectDetector::LoadModel(const std::string& model_dir, bool use_gpu) {
  paddle::AnalysisConfig config;
  config.SetModel(model_dir + "/__model__",
                  model_dir + "/__params__");
  if (use_gpu) {
      config.EnableUseGpu(100, 0);
  } else {
      config.DisableGpu();
  }
  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  // Memory optimization
  config.EnableMemoryOptim();
  predictor_ = std::move(CreatePaddlePredictor(config));
}


// Visualiztion MaskDetector results
void VisualizeResult(const cv::Mat& img,
                     const std::vector<ObjectResult>& results,
                     cv::Mat* vis_img) {
}



void ObjectDetector::Preprocess(const cv::Mat& image_mat, float shrink) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = image_mat.clone();
  cv::resize(im, im, cv::Size(), shrink, shrink, cv::INTER_CUBIC);
  im.convertTo(im, CV_32FC3, 1.0);
  int rc = im.channels();
  int rh = im.rows;
  int rw = im.cols;
  input_shape_ = {1, rc, rh, rw};
  input_data_.resize(1 * rc * rh * rw);
  float* buffer = input_data_.data();
  NormalizeImage(config_.mean, config_.scale, im, input_data_.data());
}

void ObjectDetector::Postprocess(
    const cv::Mat& raw_mat,
    float shrink,
    std::vector<ObjectResult>* result) {
  result->clear();
  int rect_num = 0;
  int rh = input_shape_[2];
  int rw = input_shape_[3];
  int total_size = output_data_.size() / 6;
  for (int j = 0; j < total_size; ++j) {
    // Class id
    int class_id = static_cast<int>(round(output_data_[0 + j * 6]));
    // Confidence score
    float score = output_data_[1 + j * 6];
    int xmin = (output_data_[2 + j * 6] * rw) / shrink;
    int ymin = (output_data_[3 + j * 6] * rh) / shrink;
    int xmax = (output_data_[4 + j * 6] * rw) / shrink;
    int ymax = (output_data_[5 + j * 6] * rh) / shrink;
    int wd = xmax - xmin;
    int hd = ymax - ymin;
    if (score > threshold_) {
      ObjectResult result_item;
      result_item.rect = {xmin, xmax, ymin, ymax};
      result_item.class_id = class_id;
      result_item.confidence = score;
      result->push_back(result_item);
    }
  }
}

void ObjectDetector::Predict(const cv::Mat& im,
                                  std::vector<ObjectResult>* result,
                                  float shrink) {
  // Preprocess image
  Preprocess(im, shrink);
  // Prepare input tensor
  auto input_names = predictor_->GetInputNames();
  auto in_tensor = predictor_->GetInputTensor(input_names[0]);
  in_tensor->Reshape(input_shape_);
  in_tensor->copy_from_cpu(input_data_.data());
  // Run predictor
  predictor_->ZeroCopyRun();
  // Get output tensor
  auto output_names = predictor_->GetOutputNames();
  auto out_tensor = predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = out_tensor->shape();
  // Calculate output length
  int output_size = 1;
  for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
  }
  output_data_.resize(output_size);
  out_tensor->copy_to_cpu(output_data_.data());
  // Postprocessing result
  Postprocess(im, shrink, result);
}
