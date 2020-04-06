#pragma once

#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
namespace PaddleDetection {
namespace preprocess {

class Normalize{
  bool is_channel_first;
  bool is_scale;
  vector<float> mean;
  vector<float> std;
  public:
    void init(bool _is_channel_first, bool _is_scale, vector<float> _mean, vector<float> _std){
        mean = _mean;
        is_channel_first = _is_channel_first;
        is_scale = _is_scale;
        std = _std;
        }
    void Run(const cv::Mat& im, cv::Mat* normalize_im);
};

class Permute{
  bool to_bgr; 
  bool channel_first;
  public:
    void init(bool _to_bgr, bool _channel_first){
        to_bgr = _to_bgr;
        channel_first = _channel_first;
        }
    void Run(const cv::Mat& im, cv::Mat* permute_im);
};

class Resize{
  std::string arch;
  int interp;
  int max_size;
  int target_size;
  bool use_cv2;
  vector<int> image_shape;
  public:
    void init(std::string _arch, int _max_size,int _target_size, int _interp, bool _use_cv2, vector<int> _image_shape){
      arch = _arch;
      interp = _interp;
      max_size = _max_size;
      target_size = _target_size;
      use_cv2 = _use_cv2;
      image_shape = _image_shape;
    }
    void Run(const cv::Mat& im, cv::Mat* resize_im);
    void GenerateScale(const cv::Mat& im, float* im_scale_x, float* im_scale_y);

};

struct PreprocessOps{
  Resize resize;
  Permute permute;
  Normalize normalize;
};
void ParsePreprocessInfo(const YAML::Node& node, std::string arch, PreprocessOps* preprocess_ops); 
}  // namespace preprocess
}  // namespace PaddleDetection
