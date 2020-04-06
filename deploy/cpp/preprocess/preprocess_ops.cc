#include <vector>
#include <string>
#include "preprocess_ops.h"

using namespace std;
namespace PaddleDetection {
namespace preprocess {

void Normalize::Run(
    const cv::Mat& im, cv::Mat* resize_im){
}

void Resize::Run(
  const cv::Mat& im, 
  cv::Mat* resize_im) {
  float im_scale_x;
  float im_scale_y;
  GenerateScale(im, &im_scale_x, &im_scale_y);
  cv::resize(im, *resize_im, cv::Size(), im_scale_x, im_scale_y, interp);
}

void Resize::GenerateScale(const cv::Mat& im, float* im_scale_x, float* im_scale_y){
  int origin_w = im.cols;
  int origin_h = im.rows;
  if (max_size != 0 && (arch == "RCNN" or arch == "RetinaNet")){
      int im_size_max = std::max(origin_w, origin_w);
      int im_size_min = std::min(origin_w, origin_w);
      float scale_ratio = static_cast<float>(target_size)
                             / static_cast<float>(im_size_min);
      if (max_size > 0) {
        if (round(scale_ratio * im_size_max) > max_size) {
                    scale_ratio = static_cast<float>(max_size)
                                / static_cast<float>(im_size_max);
                }
            }
      *im_scale_x = scale_ratio;
      *im_scale_y = scale_ratio;
  }else{
       *im_scale_x = float(target_size) / float(origin_w);
       *im_scale_y = float(target_size) / float(origin_h);
  }
     
}

void ParsePreprocessInfo(const YAML::Node& node, std::string arch, PreprocessOps* preprocess_ops) {
        for (const auto& item : node) {
           if (item["type"].as<std::string>() == "Resize"){
				preprocess_ops->resize.init(arch, item["max_size"].as<int>(), item["target_size"].as<int>(), item["interp"].as<int>(), item["use_cv2"].as<bool>(),item["image_shape"].as<vector<int>>());
            }
            else if(item["type"].as<std::string>() == "Normalize"){ 
				preprocess_ops->normalize.init(item["is_channel_first"].as<bool>(),item["is_scale"].as<bool>(), item["mean"].as<vector<float>>(), item["std"].as<vector<float>>());
            }
            else if(item["type"].as<std::string>() == "Permute"){ 
				preprocess_ops->permute.init(item["to_bgr"].as<bool>(), item["channel_first"].as<bool>());
            }
        }
    }
}  // namespace preprocess
}  // namespace PaddleDetection
