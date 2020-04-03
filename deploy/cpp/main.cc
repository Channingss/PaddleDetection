#include <iostream>
#include <string>

#include "object_detector.h"

int main(int argc, char** argv)
{
    std::string model_dir(argv[1]);
    int use_gpu = std::stoi(argv[2]);
    ModelConfig config;
    ObjectDetector det(model_dir, config, use_gpu);
    return 0;
}
