#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

using namespace std;

struct Resize{
  vector<int> image_shape;
  int iterp;
  int max_size;
  int target_size;
  bool use_cv2;
};

struct Permute{
  bool to_bgr;
  bool channel_first;
};

struct Normalize{
  bool is_channel_first;
  bool is_scale;
  vector<float> mean;
  vector<float> std;
};

struct Preprocessor{
  Resize resize;
  Permute permute;
  Normalize normalize;
};

class PaddleModelConfigPaser {
 public:
    PaddleModelConfigPaser(){}

    ~PaddleModelConfigPaser() {}

    void reset() {
    }

    bool parse_preprocessor(const YAML::Node& node, Preprocessor* preprocessor) {
        for (const auto& item : node) {
           if (item["type"].as<std::string>() == "Resize"){
				preprocessor->resize.image_shape = item["image_shape"].as<vector<int>>();
				preprocessor->resize.target_size = item["target_size"].as<int>();
				preprocessor->resize.max_size = item["max_size"].as<int>();
				preprocessor->resize.use_cv2 = item["use_cv2"].as<bool>();
            }
            else if(item["type"].as<std::string>() == "Normalize"){ 
				preprocessor->normalize.is_channel_first = item["is_channel_first"].as<bool>();
				preprocessor->normalize.is_scale = item["is_scale"].as<bool>();
				preprocessor->normalize.mean = item["mean"].as<vector<float>>();
				preprocessor->normalize.std = item["std"].as<vector<float>>();
            }
            else if(item["type"].as<std::string>() == "Permute"){ 
				preprocessor->permute.to_bgr = item["to_bgr"].as<bool>();
				preprocessor->permute.channel_first = item["channel_first"].as<bool>();
            }
        }
        return 0;
    }

    bool load_config(const std::string& conf_file) {
        reset();
        YAML::Node config;
        try {
            config = YAML::LoadFile(conf_file);
        } catch(...) {
            return false;
        }

        // 1. get 
        if (config["mode"].IsDefined()) {
            _mode = config["mode"].as<std::string>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }

        // 1. get 
        if (config["arch"].IsDefined()) {
            _arch = config["arch"].as<std::string>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }

        // 1. get 
        if (config["min_subgraph_size"].IsDefined()) {
            _min_subgraph_size = config["min_subgraph_size"].as<int>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }
        // 1. get 
        if (config["draw_threshold"].IsDefined()) {
            _draw_threshold = config["draw_threshold"].as<float>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }
        // 1. get 
        if (config["with_background"].IsDefined()) {
            _with_background = config["with_background"].as<bool>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }

        // 1. get pre_processor
        if (config["Preprocess"].IsDefined()) {
            YAML::Node preprocess_node = config["Preprocess"];
 			parse_preprocessor(preprocess_node, &_preprocessor);
             
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }

        // 1. get 
        if (config["label_list"].IsDefined()) {
            _label_list = config["label_list"].as<vector<std::string>>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }
        return true;
    }
    std::string _mode;
    float _draw_threshold;
    std::string _arch;
    int _min_subgraph_size;
    bool _with_background;
    Preprocessor _preprocessor;
    vector<std::string> _label_list;

};


int main(int argc,char** argv)
{
  PaddleModelConfigPaser config;
  config.load_config("../infer_cfg.yml");
  std::cout << config._with_background << std::endl;
  std::cout << config._label_list[0] << std::endl;
  std::cout << config._preprocessor.resize.target_size << std::endl;
  return 0;
}
