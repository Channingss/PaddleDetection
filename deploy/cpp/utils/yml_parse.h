#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include "utils.h"
#pragma once

using namespace std;
namespace PaddleDetection {
namespace utils {

class ConfigPaser {
 public:
    ConfigPaser(){}

    ~ConfigPaser() {}

    void reset() {
    }

    bool load_config(const std::string& model_dir) {
        reset();
        YAML::Node config;
        std::string conf_file;
        try {
            conf_file = path_join(model_dir, "infer_cfg.yml");
            config = YAML::LoadFile(conf_file);
        } catch(...) {
            return false;
        }

        // 1. get 
        if (config["mode"].IsDefined()) {
            mode = config["mode"].as<std::string>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }

        // 1. get 
        if (config["arch"].IsDefined()) {
            arch = config["arch"].as<std::string>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }

        // 1. get 
        if (config["min_subgraph_size"].IsDefined()) {
            min_subgraph_size = config["min_subgraph_size"].as<int>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }
        // 1. get 
        if (config["draw_threshold"].IsDefined()) {
            draw_threshold = config["draw_threshold"].as<float>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }
        // 1. get 
        if (config["with_background"].IsDefined()) {
            with_background = config["with_background"].as<bool>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }

        // 1. get pre_processor
        if (config["Preprocess"].IsDefined()) {
            preprocess_info = config["Preprocess"];
             
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }

        // 1. get 
        if (config["label_list"].IsDefined()) {
            label_list = config["label_list"].as<vector<std::string>>();
        } else {
            std::cerr << "Please set " << std::endl;
            return false;
        }
        return true;
    }
    std::string mode;
    float draw_threshold;
    std::string arch;
    int min_subgraph_size;
    bool with_background;
    YAML::Node preprocess_info;
    vector<std::string> label_list;
};

}  // namespace utils
}  // namespace PaddleDetection

