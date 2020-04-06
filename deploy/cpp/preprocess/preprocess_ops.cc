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
}

vector<float> Resize::GenerateScale(const cv::Mat& im){
  int origin_w = im.cols;
  int origin_h = im.rows;
  if (max_size != 0){
  }
}

void ParsePreprocessInfo(const YAML::Node& node, PreprocessOps* preprocess_ops) {
        for (const auto& item : node) {
           if (item["type"].as<std::string>() == "Resize"){
				preprocess_ops->resize.init(item["max_size"].as<int>(), item["target_size"].as<int>(), item["interp"].as<int>(), item["use_cv2"].as<bool>(),item["image_shape"].as<vector<int>>());
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
